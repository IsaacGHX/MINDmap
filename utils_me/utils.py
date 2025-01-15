import torch, math


def get_diagonal_indices(grid_size):
    diagonals = []
    # First half: from bottom-left to top-right
    for i in range(2 * grid_size - 1):
        diag = []
        if i < grid_size:
            # Start from the bottom row, move upwards
            start_row = i
            start_col = 0
        else:
            # Start from the rightmost column, move leftwards
            start_row = grid_size - 1
            start_col = i - grid_size + 1

        while start_row >= 0 and start_col < grid_size:
            diag.append((start_row, start_col))
            start_row -= 1
            start_col += 1
        diagonals.append(diag)

    # Assign linear sequence indices
    linear_positions = []
    global_counter = 0
    for diag in diagonals:
        line = []
        for (r, c) in diag:
            line.append((r, c, global_counter))
            global_counter += 1
        linear_positions.append(line)
    """
    [[(h, w, seq_pos), ...], ...]
    seq_pos START FROM 0!!!
    h: the latent block height
    w: the latent block width
    seq_pos: sequence in the diagonal separation
    """

    return linear_positions


def generate_reverse_mapping(diagonal_indices):
    """
    Generate a mapping from seq_pos to (row, col) using the diagonal traversal.

    Args:
        grid_size: The size of the 2D grid (grid_size x grid_size)
    Returns:
        A dictionary mapping seq_pos to (row, col)
    """
    # Create the reverse mapping
    reverse_mapping = {}
    for diag in diagonal_indices:
        for row, col, seq_pos in diag:
            reverse_mapping[seq_pos] = (row, col)

    return reverse_mapping


def create_diagonal_causal_mask(max_seq_length, max_batch_size, block_size, mode="diagonal", verbose=True):
    """
    CUDA Version =w=/
    max_seq = prefix_condition + image_latent_block
    prefix should be causal while the image block should be left-up in order of diagonal
    """
    if verbose:
        print("Creating diagonal causal mask...")
    grid_size = int(block_size ** 0.5)
    assert grid_size * grid_size == block_size
    assert max_seq_length >= block_size

    prefix_length = max_seq_length - block_size

    # batch sequence in direction of diagonal
    linear_positions = get_diagonal_indices(grid_size)

    causal_mask = torch.zeros((max_seq_length, max_seq_length), dtype=torch.bool, device='cuda')

    """prefix dealing"""
    for i in range(prefix_length):
        causal_mask[i, :i + 1] = True
    causal_mask[prefix_length:, :prefix_length] = True

    """latent content dealing"""
    if mode == "diagonal":
        """Implement Method"""
        # generate all positions
        all_positions = torch.tensor([pos for diag in linear_positions for pos in diag], device='cuda')

        # Vectorize and Broadcast
        rows = all_positions[:, 0]
        cols = all_positions[:, 1]
        seq_pos = all_positions[:, 2]

        for i in range(len(all_positions)):
            mask_condition = (
                    (rows[:i + 1] <= rows[i]) &
                    (cols[:i + 1] <= cols[i])
            )

            valid_indices = torch.where(mask_condition)[0]
            causal_mask[prefix_length + seq_pos[i], prefix_length + seq_pos[valid_indices]] = True


    elif mode == "row-flatten":
        """Method for viewing"""
        rows = torch.arange(block_size, device='cuda') // grid_size
        cols = torch.arange(block_size, device='cuda') % grid_size

        for i in range(block_size):
            current_row = rows[i]
            current_col = cols[i]

            left_condition = (cols <= current_col)
            up_condition = (rows <= current_row)

            causal_mask[prefix_length + i, prefix_length + (left_condition & up_condition)] = True

    causal_mask = causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1)
    if verbose:
        print("Diagonal causal mask created!")
    return causal_mask


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    mask = create_diagonal_causal_mask(4, 1, 4)
    print(mask.shape, mask[0])
    # 假设已经有了上面的函数定义，以下是可视化部分
    max_batch_size = 1
    prefix = 5
    block_size = 1024  # 示例的块大小，同样可按需调整
    grid_size = int(1024 ** 0.5)
    assert grid_size ** 2 == block_size
    max_seq_length = prefix + block_size
    mode = "diagonal"  # 或者 "row-flatten"，选择对应的模式
    # mode = "row-flatten"  # 或者 "row-flatten"，选择对应的模式

    causal_mask = create_diagonal_causal_mask(max_seq_length, max_batch_size, block_size, mode)
    # 将torch的bool张量转为numpy数组，并转换数据类型为适合可视化的类型（比如uint8）
    numpy_mask = causal_mask[0, :16, :16].cpu().numpy().astype(np.uint8)
    # 将False值（0）映射为黑色，True值（1）映射为白色
    print(numpy_mask.shape)
    visualization_mask = np.where(numpy_mask == 0, 0, 255)

    # 计算True的比例
    true_count = np.sum(numpy_mask == 1)
    total_elements = numpy_mask.size
    true_ratio = true_count / total_elements
    mask_causal_ratio = (0.5 - true_count / block_size ** 2) / 0.5

    print(f"True的占总体的比例: {true_ratio:.4f}, mask占据的比例: {mask_causal_ratio:.4f}")

    plt.imshow(visualization_mask, cmap='gray')
    plt.title("Causal Mask Visualization")
    plt.show()

    # Create a grid to store the sequence positions
    grid = np.zeros((grid_size, grid_size), dtype=int)
    positions = get_diagonal_indices(grid_size)

    # Fill the grid with sequence positions
    for diag in positions:
        for (r, c, seq_pos) in diag:
            grid[r, c] = seq_pos

    # Plot the grid
    fig, ax = plt.subplots()
    cax = ax.matshow(grid, cmap='plasma')

    # Annotate each cell with its sequence position
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, "", va='center', ha='center', color='white')

    # Add a color bar for reference on the right
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Generation Order')

    # Set labels and title
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))
    ax.set_xlabel('W')
    ax.set_ylabel('H')
    # ax.set_title('Generation Order Visualization')

    plt.show()
