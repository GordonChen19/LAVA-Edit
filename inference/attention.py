import torch
import torch.nn.functional as F

def toggle_save_attn(model, save_attn=True):
    for i, _ in enumerate(model.blocks):
        model.blocks[i].cross_attn.save_attn = save_attn


def get_cross_attn_map(model, token_ids):
    """
    Output avg_map: [B, L1, len(token_ids)]
    """

    block_attn = []
    for i, _ in enumerate(model.blocks):
        attn_map = model.blocks[i].cross_attn._attn_map
        block_attn.append(attn_map)

    avg_attn_map = torch.stack(block_attn, dim=0).mean(dim=0)  #Shape [B, L1, L2]
    return avg_attn_map[..., token_ids]  #Shape [B, L1, 1]


def get_attn_mask(attn_map, threshold): 
    """
    Input attn_map: [B, L1, len(token_ids)]
    Output attn_mask: [B, L1], float tensor with 0/1 values
    """  
    #Normalize attention map to [0, 1]
    attn_min = attn_map.min(dim=-1, keepdim=True)[0]
    attn_max = attn_map.max(dim=-1, keepdim=True)[0]
    attn_norm = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)

    attn_agg = attn_norm.mean(dim=-1)
    return (attn_agg > threshold).float()



#This function is only for visualization.
def unpatchify_attention(attn_map, grid_sizes):
    """
    Input attention map [B, L1] 
    Output attention map [B, F', H', W'].
    """
    B, L1 = attn_map.shape
    attn_unpatched = []

    for i in range(B):
        Fp, Hp, Wp = grid_sizes[i].tolist()
        assert Fp * Hp * Wp == L1, f"Mismatch: L1={L1}, grid product={Fp*Hp*Wp}"
        # Reshape flattened tokens back to grid
        attn_i = attn_map[i].reshape(Fp, Hp, Wp)
        attn_unpatched.append(attn_i.cpu())

    return attn_unpatched



