def plotsave_smooth_heatmap(
        df,
        filesavename,
        x_range=(0, 1),
        y_range=(0, 1),
        upscale=200,
        sigma=3,
        cmap_name='jet',
        lighten_edges=True,
        ylabel = r"Expected degree $E[D]$",
        xlabel = r"Noise amplitude $\alpha$",
        colorbar_label='Precison',
):
    """
    输入一个二维矩阵 data，绘制平滑热力图。

    参数：
        data: 2D numpy array
        x_range, y_range: tuple，定义坐标轴范围
        upscale: int，插值分辨率
        sigma: 高斯滤波平滑程度
        cmap_name: 颜色映射名称
        lighten_edges: 是否将 cmap 两端调浅
        title, xlabel, ylabel, colorbar_label: 图标题和标签
    """

    # Step 1: 提取数据和坐标
    data = df.values
    x = df.columns.values.astype(float)
    y = df.index.values.astype(float)

    # Step 2: 构建插值函数
    interp_func = RegularGridInterpolator((y, x), data, method='linear')

    # Step 3: 构建新网格
    xnew = np.linspace(x.min(), x.max(), upscale)
    ynew = np.linspace(y.min(), y.max(), upscale)
    xgrid, ygrid = np.meshgrid(xnew, ynew)
    coords = np.stack([ygrid, xgrid], axis=-1)
    data_interp = interp_func(coords)

    # Step 4: 高斯滤波
    data_smooth = gaussian_filter(data_interp, sigma=sigma)

    # Step 5: 自定义 colormap（柔化两端）
    base_cmap = cm.get_cmap(cmap_name)
    colors_map = base_cmap(np.linspace(0, 1, 256))
    if lighten_edges:
        colors_map[0] = [0.8, 0.9, 1, 1]  # 浅蓝
        colors_map[-1] = [1, 0.8, 0.8, 1]  # 浅红
    custom_cmap = colors.ListedColormap(colors_map)

    # Step 6: 绘图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data_smooth,
        cmap=custom_cmap,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect='auto'
    )
    cbar = plt.colorbar(im, label=colorbar_label)

    plt.xlabel(xlabel if xlabel else df.columns.name or 'X')
    plt.ylabel(ylabel if ylabel else df.index.name or 'Y')
    plt.title(title)
    plt.tight_layout()
    plt.show()
