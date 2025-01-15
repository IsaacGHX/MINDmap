import matplotlib.pyplot as plt

def visualize_attention_maps(attention_maps):
    """
    可视化所有层和所有头的注意力图
    :param attention_maps: 注意力分数列表，形状为 [n_layer, n_head, seqlen, seqlen]
    """
    n_layer = len(attention_maps)
    n_head = attention_maps[0].shape[1]  # 每层的头数
    seqlen = attention_maps[0].shape[2]  # 序列长度

    for layer_idx in range(n_layer):
        for head_idx in range(n_head):
            # 获取当前层和头的注意力图
            attn_map = attention_maps[layer_idx][0, head_idx].cpu().numpy()  # [seqlen, seqlen]

            # 绘制热力图
            plt.figure(figsize=(10, 10))
            plt.imshow(attn_map, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f"Layer {layer_idx} Head {head_idx} Attention Map")
            plt.xlabel("Key Tokens")
            plt.ylabel("Query Tokens")
            plt.show()