import os
import random

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

# Data
if not os.path.exists("data.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "data.txt")

with open("data.txt") as file:
    names = [line.strip() for line in file if line.strip()]

chars = sorted(set("".join(names)))  # Por qué hace falta sorted?
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

BOS = len(chars)
vocab_size = len(chars) + 1

# Hyperparameters
n_embd = 64
n_head = 8
head_dim = n_embd // n_head
n_layer = 4
block_size = 16  # longitud máxima de la secuencia de tokens. No es el batch size


# Model
class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        # self.kv_cache = torch.Tensor(size=(block_size,))

    def rmsnorm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)

    def forward(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ):
        """
        x:
            training: (B, T, C)
            cached inference: usually (B, 1, C)

        kv_cache:
            None
                or
            (cached_k, cached_v)

        cached_k/v shapes:
            (B, H, T_cached, D)
        """
        B, T, C = (
            x.shape
        )  # (batch, seq_len == nº nuevos tokens a procesar, d_emb_space)
        head_size = C // n_head

        # attention
        h = self.rmsnorm(x)

        q: torch.Tensor = self.wq(h)
        k: torch.Tensor = self.wk(h)
        v: torch.Tensor = self.wv(h)

        # Explicación: tenemos que seccionar cada Q, K y V para hacer el attention por cada head
        #   - creamos tensores nuevos con cada slice concatenado para evitar iterar
        #   - El orden del view es así porque los números se guardan iterando primero por
        #   el último índice (o sea, row-major). Esto significa que al cambiar C --> (n_head, head_size)
        #   le estamos diciendo divide en n_head bloques de tamaño head_size. Lo contrario no tendría
        #   sentido (aunque las dimensiones se pueden hacer cuadrar también).
        #   - El transpose hace falta porque el matmul (u operador @) usa las últimas dos dimensiones y
        #   trata al resto como "batch size". Nosotros queremos multiplicar longitud de secuencia por el
        #   dimensión del espacio embebido, que dentro de cada head es el head_size.
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        # if we have cache, we are doing inference and T = 1
        if kv_cache is not None:
            cached_k, cached_v = kv_cache

            # concatenate along T (sequence) axis
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)

        # para calcular la atención tenemos que transponer K. tras la transposición original para poder
        # multiplicar, aquí las dos últimas dimensiones son las que nos importan y el resto tienen carácter
        # de "batch size".
        attn_logits = (q @ k.transpose(-2, -1)) / head_dim**0.5

        # We only need to mask when processing several tokens at once
        if kv_cache is None:
            # usamos una máscara para eliminar la atención a tokens futuros
            # en este punto, cada (batch x head) tendrá unos attn_logits de tamaño (T x T)
            # --> estamos multiplicando (T x H) · (H x T) --> (T x T)
            T_total = k.size(2)

            # Aquí se usa T_total aunque si no hay kv_cache T_total será igual T_new
            # Es decir, el nº de nuevos tokens a procesar será igual al total de tokens
            # Sin embargo, conceptualmente son diferentes, aunque su valor coincida si kv_cache es None
            mask = torch.tril(
                torch.ones(T_total, T_total, device=x.device, dtype=torch.bool)
            )
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)

        x = x + y

        # MLP
        h = self.rmsnorm(x)
        y = self.fc1(h)
        y = torch.relu(y)
        y = self.fc2(y)
        x = x + h

        return x, new_cache


class MicroGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # token and positional embeddings
        # -> Qué diferencia hay entre definir esto como un Tensor random?
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)

        # Attention
        self.attn_blocks = nn.ModuleList([AttnBlock() for _ in range(n_layer)])

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def rmsnorm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)

    def forward(
        self,
        idx: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        """
        idx:
            training:           (B, T)
            cached inference:   usually (B, 1)

        kv_cache:
            None
                or
            [
                (k_layer0, v_layer0),
                (k_layer1, v_layer1),
                ...
            ]
        """
        B, T = idx.shape

        if kv_cache is None:
            past_len = 0
        else:
            past_len = kv_cache[0][0].size(2)

        # If there is kv cache, esto valdrá por ejemplo (5, 6), del índice anterior al actual.
        positions = torch.arange(past_len, past_len + T, device=idx.device)

        tok = self.wte(idx)
        pos = self.wpe(positions)

        x = tok + pos

        new_cache = []

        for i, block in enumerate(self.attn_blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]

            x, layer_new_cache = block(x, kv_cache=layer_cache)

            new_cache.append(layer_new_cache)

        x = self.rmsnorm(x)

        return self.lm_head(x), new_cache


if __name__ == "__main__":
    # B, T, C = 2, 4, 16
    batch_size = 32
    B, T, C = batch_size, block_size, n_embd

    # # Testear el attention block
    # block = AttnBlock()
    # dummy_x = torch.randn(B, T, C)
    # block(dummy_x)

    # # Testear el modelo completo
    # model = MicroGPT()
    # dummy_x = torch.randint(
    #     0, vocab_size - 1, (B, T)
    # )  # <-- Tienen que ser enteros dentro del rango del vocabulario
    # model(dummy_x)

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MicroGPT().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def get_batch(batch_size):
        xs = []
        ys = []

        for _ in range(batch_size):
            name = random.choice(names)

            tokens = [BOS] + [stoi[c] for c in name] + [BOS]

            x = tokens[:-1]
            y = tokens[1:]

            # recortamos la secuencia si llega al fin de la ventana (nombre demasiado largo por ejemplo)
            x = x[:block_size]
            y = y[:block_size]

            # añadimos padding usando el propio BOS
            pad = block_size - len(x)
            x = x + [BOS] * pad
            y = y + [BOS] * pad

            xs.append(x)
            ys.append(y)

        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)

        return x, y

    num_steps = 5_000
    progress_bar = trange(num_steps)
    for step in progress_bar:
        x, y = get_batch(batch_size)

        # breakpoint()
        # print(x.shape)
        # print(y.shape)

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Inference
    @torch.no_grad()
    def generate(
        model,
        num_samples=10,
        temperature=1.0,
    ):
        model.eval()
        results = []

        for _ in range(num_samples):
            kv_cache = None

            # Start with BOS
            token = torch.tensor([[BOS]], dtype=torch.long, device=device)

            chars = []

            for _ in range(block_size):
                logits, kv_cache = model(token, kv_cache=kv_cache)

                # Only one token was passed, so position -1
                # is the prediction for the next token
                logits = logits[:, -1, :]

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())

                if token_id == BOS:
                    break

                chars.append(itos[token_id])

                # Crucial:
                # next iteration receives ONLY the new token.
                token = next_token

            results.append("".join(chars))

        return results

    hallucinated_names = generate(model, num_samples=20, temperature=1)

    for name in hallucinated_names:
        print(name)

    # TODO: check that the new names are not in the training set