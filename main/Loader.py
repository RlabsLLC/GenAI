import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import sys
import os

try:
    import gguf
except ImportError:
    print("[!] Please run 'pip install gguf torch' to run this model.")
    sys.exit(1)

# --- Fixed Architecture Settings ---
# These must match the original training script
block_size = 32
n_head = 4
device = 'cpu'

# --- 1. Rebuild the Brain Architecture ---
class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # No dropout during inference
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = OptimizedMultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GenA1(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- 2. Load Model and Chat ---
def load_and_chat():
    print("\n--- GenA1 Loader ---")
    
    # Ask the user which model to load
    model_name = input("Enter the path to the .gguf file (e.g., GenA1_250M_params.gguf): ").strip()
    if not os.path.exists(model_name):
        print(f"[!] Could not find {model_name}.")
        return

    # Find the matching vocabulary file
    vocab_name = model_name.replace(".gguf", "_vocab.json")
    if not os.path.exists(vocab_name):
        print(f"[!] Could not find vocabulary file: {vocab_name}. Make sure it is in the same folder!")
        return

    # Load the alphabet
    with open(vocab_name, "r") as f:
        chars = json.load(f)
    
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, stoi.get(' ', 0)) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print("\n[+] Loading brain waves from GGUF...")
    reader = gguf.GGUFReader(model_name)
    
    # Read tensors into PyTorch format
    state_dict = {}
    for tensor in reader.tensors:
        # Convert half-precision (FP16) back to standard float if necessary
        if tensor.data.dtype == 'float16':
            tensor_data = torch.tensor(tensor.data.astype('float32'))
        else:
            tensor_data = torch.tensor(tensor.data)
        state_dict[tensor.name] = tensor_data

    # Dynamically figure out how big the model grew!
    n_embd = state_dict['token_embedding_table.weight'].shape[1]
    
    # Count how many layers it has by checking the dictionary keys
    layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]
    n_layer = max(layer_indices) + 1 if layer_indices else 0

    print(f"[+] Rebuilding custom architecture: {n_layer} layers, {n_embd} embedding width.")
    
    # Build the model and load the weights
    model = GenA1(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer)
    model.load_state_dict(state_dict)
    model.eval() # Set to inference mode

    print("\n=====================================================")
    print("GenA1 is loaded and ready! Type 'quit' to exit.")
    print("=====================================================\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            # Safely encode input, replacing unknown characters with spaces
            safe_input = ''.join([c if c in stoi else ' ' for c in user_input])
            context = torch.tensor([encode(safe_input)], dtype=torch.long, device=device)
            
            print("GenA1: ", end="", flush=True)
            
            # Generate up to 200 tokens
            generated_indices = model.generate(context, max_new_tokens=200)[0].tolist()
            new_indices = generated_indices[len(context[0]):]
            
            print(decode(new_indices))
            print("-" * 30)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    load_and_chat()