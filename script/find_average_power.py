import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path, channel=0):
    """Load data and explain the dimensions"""
    stack = tifffile.imread(file_path)
   
    data = stack[:, :, channel, :, :]
    print(f"The shape of the processed data: {data.shape} (time, Z, Y, X)")
    
    return data

def extract_center_region_3d(volume, center_ratio=0.3):
    """Extract the central region of the 3D volume"""
    depth, height, width = volume.shape
    center_d, center_h, center_w = depth // 2, height // 2, width // 2
    
    crop_depth = int(depth * center_ratio)
    crop_height = int(height * center_ratio)
    crop_width = int(width * center_ratio)
    
    start_d = max(0, center_d - crop_depth // 2)
    start_h = max(0, center_h - crop_height // 2)
    start_w = max(0, center_w - crop_width // 2)
    end_d = min(depth, start_d + crop_depth)
    end_h = min(height, start_h + crop_height)
    end_w = min(width, start_w + crop_width)
    
    center_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
    
    return center_volume, (start_w, start_h, start_d, end_w - start_w, end_h - start_h, end_d - start_d)

# ----------------------------
# 3D Displacement and Strain Calculation
# ----------------------------
def compute_3d_displacement_complete(vol1, vol2, bbox):
    """Complete 3D displacement field calculation"""
    #Extract the cell region
    x, y, z, w, h, d = bbox
    cell_vol1 = vol1[z:z+d, y:y+h, x:x+w]
    cell_vol2 = vol2[z:z+d, y:y+h, x:x+w]
    
    
    displacement_3d_cell = np.zeros((cell_vol1.shape[0], cell_vol1.shape[1], cell_vol1.shape[2], 3), dtype=np.float32)
    
    for z_idx in range(cell_vol1.shape[0]):
        img1 = cv2.normalize(cell_vol1[z_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2 = cv2.normalize(cell_vol2[z_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        displacement_3d_cell[z_idx, :, :, 0] = flow[:, :, 0]  # x方向
        displacement_3d_cell[z_idx, :, :, 1] = flow[:, :, 1]  # y方向
    
    # Z-direction displacement estimation
    for z_idx in range(1, cell_vol1.shape[0]-1):
        disp_above = np.sqrt(displacement_3d_cell[z_idx-1, :, :, 0]**2 + displacement_3d_cell[z_idx-1, :, :, 1]**2)
        disp_below = np.sqrt(displacement_3d_cell[z_idx+1, :, :, 0]**2 + displacement_3d_cell[z_idx+1, :, :, 1]**2)
        displacement_3d_cell[z_idx, :, :, 2] = 0.5 * (disp_above + disp_below) * 0.05
    
    # Create a displacement field of the total volume size
    full_displacement_3d = np.zeros((vol1.shape[0], vol1.shape[1], vol1.shape[2], 3), dtype=np.float32)
    full_displacement_3d[z:z+d, y:y+h, x:x+w] = displacement_3d_cell
    
    return full_displacement_3d, displacement_3d_cell, (cell_vol1, cell_vol2)

def calculate_3d_strain_complete(displacement_3d_cell):
    """Complete 3D strain field calculation"""
    # Enlarge the displacement field to facilitate better calculation of the gradient
    displacement_scaled = displacement_3d_cell * 100.0
    
    # Calculate 3D gradient
    gradients = []
    for i in range(3):  
        grad_z, grad_y, grad_x = np.gradient(displacement_scaled[..., i])
        gradients.append((grad_x, grad_y, grad_z))
    
    # Calculate the 3D strain tensor
    epsilon_xx = gradients[0][0] / 100.0  # du/dx
    epsilon_yy = gradients[1][1] / 100.0  # dv/dy
    epsilon_zz = gradients[2][2] / 100.0  # dw/dz
    epsilon_xy = 0.5 * (gradients[0][1] + gradients[1][0]) / 100.0  # 0.5*(du/dy + dv/dx)
    epsilon_xz = 0.5 * (gradients[0][2] + gradients[2][0]) / 100.0  # 0.5*(du/dz + dw/dx)
    epsilon_yz = 0.5 * (gradients[1][2] + gradients[2][1]) / 100.0  # 0.5*(dv/dz + dw/dy)
    
    # Smooth strain field
    epsilon_xx = ndimage.gaussian_filter(epsilon_xx, sigma=1)
    epsilon_yy = ndimage.gaussian_filter(epsilon_yy, sigma=1)
    epsilon_zz = ndimage.gaussian_filter(epsilon_zz, sigma=1)
    epsilon_xy = ndimage.gaussian_filter(epsilon_xy, sigma=1)
    epsilon_xz = ndimage.gaussian_filter(epsilon_xz, sigma=1)
    epsilon_yz = ndimage.gaussian_filter(epsilon_yz, sigma=1)
    
    strain_3d = np.array([
        [epsilon_xx, epsilon_xy, epsilon_xz],
        [epsilon_xy, epsilon_yy, epsilon_yz],
        [epsilon_xz, epsilon_yz, epsilon_zz]
    ])
    
    return strain_3d

# ----------------------------
# Stress Calculation Module
# ----------------------------
def calculate_3d_stress_cell_region(strain_3d):
    """Calculate the 3D stress field (cellular region) based on strain analysis"""
   

    E = 28.0  
    nu = 0.25  
    
    mu = E / (2 * (1 + nu))
    lambda_ = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))  # 拉梅第一常数
   
    d, h, w = strain_3d[0, 0].shape
    
    # Initial stress tensor
    stress_3d = np.zeros((3, 3, d, h, w), dtype=np.float32)
    
     # Extract the strain components
    epsilon_xx = strain_3d[0, 0]
    epsilon_yy = strain_3d[1, 1]
    epsilon_zz = strain_3d[2, 2]
    epsilon_xy = strain_3d[0, 1]
    epsilon_xz = strain_3d[0, 2]
    epsilon_yz = strain_3d[1, 2]
    
    # Calculate the volumetric strain
    volumetric_strain = epsilon_xx + epsilon_yy + epsilon_zz
    
    # Calculate the components of normal stress (using the complete formula)
    stress_3d[0, 0] = lambda_ * volumetric_strain + 2 * mu * epsilon_xx  # σ_xx
    stress_3d[1, 1] = lambda_ * volumetric_strain + 2 * mu * epsilon_yy  # σ_yy
    stress_3d[2, 2] = lambda_ * volumetric_strain + 2 * mu * epsilon_zz  # σ_zz
    
    #
    stress_3d[0, 1] = 2 * mu * epsilon_xy  # σ_xy
    stress_3d[1, 0] = 2 * mu * epsilon_xy  # σ_yx 
    
    stress_3d[0, 2] = 2 * mu * epsilon_xz  # σ_xz
    stress_3d[2, 0] = 2 * mu * epsilon_xz  # σ_zx 
    
    stress_3d[1, 2] = 2 * mu * epsilon_yz  # σ_yz
    stress_3d[2, 1] = 2 * mu * epsilon_yz  # σ_zy 
    
    
    return stress_3d, mu

def calculate_traction_stress_cell(stress_3d, normal_vectors=None):
    """Calculate the traction stress τ = σ · n"""
    
    d, h, w = stress_3d[0, 0].shape
    
    
    if normal_vectors is None:
        normal_vectors = np.zeros((d, h, w, 3), dtype=np.float32)
        normal_vectors[:, :, :, 2] = 1.0  
    
    
    traction_stress = np.zeros((d, h, w, 3), dtype=np.float32)
    
  
    for i in range(3):
        for j in range(3):
            traction_stress[:, :, :, i] += stress_3d[i, j] * normal_vectors[:, :, :, j]
    
   
    traction_magnitude = np.sqrt(traction_stress[:, :, :, 0]**2 + 
                                traction_stress[:, :, :, 1]**2 + 
                                traction_stress[:, :, :, 2]**2)
    
    
    return traction_stress, traction_magnitude

# ----------------------------
# Time series stress analysis - Calculating the average value
# ----------------------------
def analyze_stress_time_series(data, center_ratio=0.4, reference_time=0):
    
    n_time, n_z, height, width = data.shape
   
    time_avg_stress = np.zeros(n_time)
    stress_time_series = []
    
    # Extract the central area at all time points
    center_volumes = []
    center_bboxes = []
    
    for t in range(n_time):
        vol_center, bbox_center = extract_center_region_3d(data[t], center_ratio)
        center_volumes.append(vol_center)
        center_bboxes.append(bbox_center)
    
    # Using the first time point as a reference, examine the cell area
    ref_vol = center_volumes[reference_time]
    x, y, z, w, h, d = 0, 0, 0, ref_vol.shape[2], ref_vol.shape[1], ref_vol.shape[0]
    bbox = (x, y, z, w, h, d)
    
 
    
    # Calculate the stress at each time point (relative to the reference time point)
    for t in range(n_time):
        if t == reference_time:
            # Reference time point: Stress is 0
            avg_stress = 0.0
            stress_time_series.append(None)
        else:
            print(f"Calculate the stress at the time point {t}...")
            
            # Calculate displacement
            displacement_full, displacement_cell, _ = compute_3d_displacement_complete(
                center_volumes[reference_time], center_volumes[t], bbox)
            
            # Calculate strain
            strain_3d = calculate_3d_strain_complete(displacement_cell)
            
            # computed stress
            stress_3d, mu = calculate_3d_stress_cell_region(strain_3d)
            
            # Calculate strain
            strain_3d = calculate_3d_strain_complete(displacement_cell)
            
            # computed stress
            stress_3d, mu = calculate_3d_stress_cell_region(strain_3d)
            s = stress_3d
            von_mises = np.sqrt(0.5 * (
            (s[0,0] - s[1,1])**2 + 
            (s[1,1] - s[2,2])**2 + 
            (s[2,2] - s[0,0])**2 +
            6 * (s[0,1]**2 + s[0,2]**2 + s[1,2]**2)
        ))
            
           
            avg_stress = np.mean(von_mises)
            
            # Store stress data
            stress_data = {
                'displacement': displacement_cell,
                'strain': strain_3d,
                'stress': stress_3d,
                'traction_magnitude': von_mises,
                'avg_stress': avg_stress  
            }
            stress_time_series.append(stress_data)
        
        
        time_avg_stress[t] = avg_stress
       
    
    return time_avg_stress, stress_time_series, center_bboxes[0]

def visualize_time_series_stress(time_avg_stress, stress_time_series):
    """Visualization of the stress analysis results of time series"""
    
    n_time = len(time_avg_stress)
    
    # Create a time series chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average stress varies with time
    time_points = np.arange(n_time)
    axes[0, 0].plot(time_points, time_avg_stress, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('time point')
    axes[0, 0].set_ylabel('Average stress (Pa)')
    axes[0, 0].set_title('Average stress varies with time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark the maximum value
    max_idx = np.argmax(time_avg_stress)
    axes[0, 0].plot(max_idx, time_avg_stress[max_idx], 'ro', markersize=10, 
                   label=f'maximum: {time_avg_stress[max_idx]:.2e} Pa')
    axes[0, 0].legend()
    
    # 2. Stress rate of change
    if n_time > 1:
        stress_growth = np.diff(time_avg_stress)
        axes[0, 1].plot(time_points[1:], stress_growth, 'go-', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('time point')
        axes[0, 1].set_ylabel('Stress rate of change (Pa/time point)')
        axes[0, 1].set_title('Stress rate of change')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. cumulative stress
    cumulative_stress = np.cumsum(time_avg_stress)
    axes[1, 0].plot(time_points, cumulative_stress, 'mo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('time point')
    axes[1, 0].set_ylabel('Cumulative stress (Pa)')
    axes[1, 0].set_title('cumulative stress')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistical analysis of stress distribution
    if any(stress_time_series):
        valid_stresses = [ts['avg_stress'] for ts in stress_time_series if ts is not None]
        axes[1, 1].hist(valid_stresses, bins=min(10, len(valid_stresses)), alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Average stress (Pa)')
        axes[1, 1].set_ylabel('frequency')
        axes[1, 1].set_title('Average stress distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    for t in range(n_time):
        if t == 0:
            change = 0.0
        else:
            change = time_avg_stress[t] - time_avg_stress[t-1]
        print(f"{t}\t{time_avg_stress[t]:.6e}\t{change:+.6e}")
    
    return fig

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    try:
        
        file_path = "D:/转移文件/吴旭东老师的求助/video/Experiment-1424.tif"
        data = load_data(file_path, channel=1)
        
        time_avg_stress, stress_time_series, center_bbox = analyze_stress_time_series(
            data, center_ratio=0.4, reference_time=0)
    
        fig_time_series = visualize_time_series_stress(time_avg_stress, stress_time_series)
        
       
       
    except Exception as e:
        print(f"find bug: {e}")
        import traceback
        traceback.print_exc()
