import torch
from torch.autograd import Variable
from gpt_multipos import GPT_B, LabelEmbedder
# 假设你已经定义好的 GPT_B 模型和相关的类
# from your_model_file import GPT_B

def test_diagonal_generation():
    # 基本参数
    batch_size = 1
    seq_size = 32 * 32

    # 初始化模型（示例）
    model = GPT_B(block_size=seq_size).to("cuda")
    model.eval()

    input_pos = torch.tensor([i - 1 for i in range(497, 529)], dtype=torch.long, device="cuda")
    # input_pos = torch.tensor([3,4,5], dtype=torch.long, device="cuda")
    # input_pos = torch.tensor([1,2], dtype=torch.long, device="cuda")
    # input_pos = None
    # 假设我们已有前面已预测好的3个token作为上下文
    # 这3个token是前两对角线预测完毕后已经生成的token数目(第0批1个+第1批2个=3个)
    # 在实际中，这些值应是模型在前两批预测中得到的输出token，这里用随机数代替
    # idx = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, len(input_pos)-1), device = "cuda")
    idx = torch.randn(size=(batch_size, len(input_pos), 768), device="cuda")

    # cond_idx假设为1个条件token，示例中也用随机值或0代替
    cond_idx = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    with torch.device("cuda"):
        model.setup_caches(batch_size, seq_size, dtype=model.tok_embeddings.weight.dtype)

    # 第3批（0-based为2号对角线），对应的三个位置下标
    # (0,2) → 2, (1,1) → 17, (2,0) → 32
    # 这三个位置需要一起预测

    # 目标tokens（ground truth），如果需要计算loss，可随机生成:
    # targets = torch.randint(low=1, high=model.config.vocab_size, size=(batch_size, len(input_pos)), device="cuda")
    targets = torch.randint(low=1, high=model.config.vocab_size, size=(batch_size, 10), device="cuda")

    # assert targets.min() >= 0 and targets.max() < model.config.num_classes, \
    #     f"Targets out of range: min={targets.min().item()}, max={targets.max().item()}"

    # valid全1，表示这三个预测位置都有效
    valid = torch.ones(batch_size, dtype=torch.float32, device="cuda")

    # 调用模型前向
    # 根据之前的实现，model.forward(
    #   idx, cond_idx, input_pos=input_pos, targets=targets, valid=valid
    # ) 会返回 (logits, loss)
    logits, loss = model.forward(
        cond_idx=None,
        idx_embedding=idx,
        input_pos=input_pos,
        # targets=targets
    )

    print("Logits shape:", logits.shape, )  # [batch_size, num_positions, vocab_size]
    if loss is not None:
        print("Loss:", loss.item())


def test_label_embedder():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 参数设置
    num_classes = 10
    hidden_size = 16
    dropout_prob = 0.5

    # 创建 LabelEmbedder 实例
    label_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob)

    # 创建输入标签 (假设有 100 个样本)
    labels = torch.randint(0, num_classes, (8,))
    force_drop_ids = torch.zeros((8,), dtype=torch.bool,)

    # 测试 token_drop 方法
    print("测试 token_drop 方法:")
    dropped_labels = label_embedder.token_drop(labels)
    print("原始标签:", labels)
    print("丢弃后的标签:", dropped_labels)
    print("被丢弃的标签数量:", (dropped_labels == num_classes).sum().item())

    # 测试 forward 方法 (训练模式)
    print("\n测试 forward 方法 (训练模式):")
    embeddings = label_embedder.forward(labels, train=True)
    print("嵌入张量形状:", embeddings.shape)  # 应为 (100, 1, hidden_size)

    # 测试 forward 方法 (推理模式)
    print("\n测试 forward 方法 (推理模式):")
    embeddings = label_embedder.forward(labels, train=False, force_drop_ids=force_drop_ids)
    print("嵌入张量形状:", embeddings.shape)  # 应为 (100, 1, hidden_size)

if __name__ == '__main__':
    # 执行测试函数
    # test_diagonal_generation()
    test_label_embedder()