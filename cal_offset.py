import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    return Rotation.from_matrix(R).as_quat()

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    return Rotation.from_quat(q).as_matrix()

def transform_error(x, A_to_C_list, B_to_C_list):
    """计算变换矩阵的误差函数，使用欧拉角和位移量的残差平方和
    Args:
        x: 优化变量，前4个是四元数，后3个是平移向量
        A_to_C_list: A到C的变换矩阵列表
        B_to_C_list: B到C的变换矩阵列表
    Returns:
        欧拉角和位移残差平方和
    """
    q = x[:4] / np.linalg.norm(x[:4])  # 归一化四元数
    t = x[4:]
    R = quaternion_to_rotation_matrix(q)
    
    # 构建A到B的变换矩阵
    A_to_B = np.eye(4)
    A_to_B[:3, :3] = R
    A_to_B[:3, 3] = t
    
    total_error = 0
    for A_to_C, B_to_C in zip(A_to_C_list, B_to_C_list):
        # 计算: A_to_C = B_to_C * A_to_B
        predicted_A_to_C = B_to_C @ A_to_B
        
        # 计算旋转误差（欧拉角）
        actual_euler = Rotation.from_matrix(A_to_C[:3, :3]).as_euler('xyz', degrees=True)
        predicted_euler = Rotation.from_matrix(predicted_A_to_C[:3, :3]).as_euler('xyz', degrees=True)
        euler_error = np.sum((actual_euler - predicted_euler) ** 2)
        
        # 计算平移误差
        translation_error = np.sum((A_to_C[:3, 3] - predicted_A_to_C[:3, 3]) ** 2)
        
        # 可以为旋转和平移误差添加权重
        w_rotation = 1.0  # 旋转误差权重
        w_translation = 1.0  # 平移误差权重
        
        total_error += w_rotation * euler_error + w_translation * translation_error
    
    return total_error

def optimize_transform(A_to_C_list, B_to_C_list):
    """优化求解A到B的变换矩阵
    Args:
        A_to_C_list: A到C的变换矩阵列表
        B_to_C_list: B到C的变换矩阵列表
    Returns:
        优化后的A到B变换矩阵
    """
    # 初始猜测：使用第一组数据的结果
    initial_A_to_B = np.linalg.inv(A_to_C_list[0]) @ B_to_C_list[0]
    initial_R = initial_A_to_B[:3, :3]
    initial_t = initial_A_to_B[:3, 3]
    initial_q = rotation_matrix_to_quaternion(initial_R)
    
    # 初始参数：四元数和平移向量
    x0 = np.concatenate([initial_q, initial_t])
    
    # 优化
    result = minimize(
        transform_error, 
        x0, 
        args=(A_to_C_list, B_to_C_list),
        method='Nelder-Mead'
    )
    
    # 从优化结果构建变换矩阵
    optimal_q = result.x[:4] / np.linalg.norm(result.x[:4])
    optimal_t = result.x[4:]
    optimal_R = quaternion_to_rotation_matrix(optimal_q)
    
    optimal_transform = np.eye(4)
    optimal_transform[:3, :3] = optimal_R
    optimal_transform[:3, 3] = optimal_t
    
    return optimal_transform

def calculate_6dof_error(true_RT, estimated_RT):
    """计算RT矩阵的6自由度误差
    Args:
        true_RT: 真实的RT矩阵
        estimated_RT: 估计的RT矩阵
    Returns:
        euler_errors: 3个旋转角度的误差（度）
        translation_errors: 3个平移分量的误差
    """
    # 计算旋转误差（欧拉角，度）
    true_euler = Rotation.from_matrix(true_RT[:3, :3]).as_euler('xyz', degrees=True)
    estimated_euler = Rotation.from_matrix(estimated_RT[:3, :3]).as_euler('xyz', degrees=True)
    euler_errors = np.abs(true_euler - estimated_euler)
    
    # 计算平移误差
    translation_errors = np.abs(true_RT[:3, 3] - estimated_RT[:3, 3])
    
    return euler_errors, translation_errors

# 示例使用
if __name__ == "__main__":
    # 生成5组测试数据
    np.random.seed(42)
    A_to_C_list = []
    B_to_C_list = []
    
    # 真实的A到B变换矩阵（用于生成测试数据）
    true_R = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    true_t = np.array([1.0, 2.0, 3.0])
    true_A_to_B = np.eye(4)
    true_A_to_B[:3, :3] = true_R
    true_A_to_B[:3, 3] = true_t
    
    # 生成5组带噪声的测试数据
    for i in range(5):
        # 随机生成B到C的变换
        B_to_C = np.eye(4)
        B_to_C[:3, :3] = Rotation.random().as_matrix()
        B_to_C[:3, 3] = np.random.rand(3) * 5
        
        # 计算对应的A到C的变换
        A_to_C = B_to_C @ true_A_to_B
        
        # 添加欧拉角噪声（角度制）
        noise_angles = (np.random.rand(3) * 2 - 1) * 2  # 生成±2度的随机噪声
        current_euler = Rotation.from_matrix(A_to_C[:3, :3]).as_euler('xyz', degrees=True)
        noisy_euler = current_euler + noise_angles
        A_to_C[:3, :3] = Rotation.from_euler('xyz', noisy_euler, degrees=True).as_matrix()
        
        # 添加平移噪声
        noise_translation = (np.random.rand(3) * 0.1 - 0.05)  # 生成±0.05的随机噪声
        A_to_C[:3, 3] += noise_translation
        
        A_to_C_list.append(A_to_C)
        B_to_C_list.append(B_to_C)
    
    # 优化求解
    estimated_A_to_B = optimize_transform(A_to_C_list, B_to_C_list)
    
    # 计算6自由度误差
    euler_errors, translation_errors = calculate_6dof_error(true_A_to_B, estimated_A_to_B)
    
    # 打印结果
    print("真实的A到B变换矩阵:")
    print(true_A_to_B)
    print("\n估计的A到B变换矩阵:")
    print(estimated_A_to_B)
    print("\n旋转误差 (度):")
    print(f"Roll (x): {euler_errors[0]:.4f}°")
    print(f"Pitch (y): {euler_errors[1]:.4f}°")
    print(f"Yaw (z): {euler_errors[2]:.4f}°")
    print("\n平移误差 (米):")
    print(f"X: {translation_errors[0]:.4f}m")
    print(f"Y: {translation_errors[1]:.4f}m")
    print(f"Z: {translation_errors[2]:.4f}m")
    
    # 计算总体误差
    total_rotation_error = np.linalg.norm(euler_errors)
    total_translation_error = np.linalg.norm(translation_errors)
    print(f"\n总旋转误差: {total_rotation_error:.4f}°")
    print(f"总平移误差: {total_translation_error:.4f}m")