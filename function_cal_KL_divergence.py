import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

def calculate_statistics(points):
    """计算点云的统计特征
    Args:
        points: numpy数组，形状为(N, D)，N是点的数量，D是维度
    Returns:
        mean: 均值
        cov: 协方差矩阵
    """
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    return mean, cov

def estimate_density(points, evaluation_points):
    """使用核密度估计计算概率密度
    Args:
        points: 用于估计密度的点云数据
        evaluation_points: 需要评估密度的位置
    Returns:
        密度估计值
    """
    kde = gaussian_kde(points.T)
    return kde(evaluation_points.T)

def kl_divergence(p, q):
    """计算KL散度
    Args:
        p: 第一个分布的概率密度
        q: 第二个分布的概率密度
    Returns:
        KL散度值
    """
    # 避免除以零，给q添加一个很小的值
    epsilon = 1e-10
    q = q + epsilon
    return np.sum(p * np.log(p / q))

def compute_cloud_statistics_and_kl(cloud1, cloud2, n_samples=1000):
    """计算两组点云的统计量和KL散度
    Args:
        cloud1: 第一组点云数据，numpy数组
        cloud2: 第二组点云数据，numpy数组
        n_samples: 用于估计KL散度的采样点数量
    Returns:
        统计量和KL散度
    """
    # 计算统计量
    mean1, cov1 = calculate_statistics(cloud1)
    mean2, cov2 = calculate_statistics(cloud2)
    
    # 生成用于估计密度的采样点
    x_min = min(cloud1.min(), cloud2.min())
    x_max = max(cloud1.max(), cloud2.max())
    eval_points = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    
    # 估计概率密度
    p = estimate_density(cloud1, eval_points)
    q = estimate_density(cloud2, eval_points)
    
    # 计算KL散度
    kl_div = kl_divergence(p, q)
    
    return {
        'cloud1_mean': mean1,
        'cloud1_cov': cov1,
        'cloud2_mean': mean2,
        'cloud2_cov': cov2,
        'kl_divergence': kl_div
    }

# 示例使用
if __name__ == "__main__":
    # 导入绘图库
    import matplotlib.pyplot as plt
    
    # 生成示例数据
    np.random.seed(42)
    cloud1 = np.random.normal(0, 1, (1000, 1))
    cloud2 = np.random.normal(0.5, 1.2, (1000, 1))
    
    # 计算统计量和KL散度
    results = compute_cloud_statistics_and_kl(cloud1, cloud2)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制点云直方图
    plt.subplot(121)
    plt.hist(cloud1, bins=30, alpha=0.5, label='Cloud 1', density=True)
    plt.hist(cloud2, bins=30, alpha=0.5, label='Cloud 2', density=True)
    plt.title('Point Cloud Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # 绘制核密度估计
    plt.subplot(122)
    x_eval = np.linspace(min(cloud1.min(), cloud2.min()),
                        max(cloud1.max(), cloud2.max()), 1000).reshape(-1, 1)
    kde1 = gaussian_kde(cloud1.T)
    kde2 = gaussian_kde(cloud2.T)
    
    plt.plot(x_eval, kde1(x_eval.T), label='Cloud 1 KDE')
    plt.plot(x_eval, kde2(x_eval.T), label='Cloud 2 KDE')
    plt.title('Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    
    # 打印统计结果
    print("Cloud 1 Statistics:")
    print(f"Mean: {results['cloud1_mean']}")
    print(f"Covariance: {results['cloud1_cov']}")
    print("\nCloud 2 Statistics:")
    print(f"Mean: {results['cloud2_mean']}")
    print(f"Covariance: {results['cloud2_cov']}")
    print(f"\nKL Divergence: {results['kl_divergence']}")
    
    plt.show()


    