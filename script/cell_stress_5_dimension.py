import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path, channel=0):
    """Load data and explain the dimensions"""
    stack = tifffile.imread(file_path)
    print(f"Shape of the original data: {stack.shape} (time, Z, channel, Y, X)")
    print(f"data type: {stack.dtype}, numerical range: [{stack.min()}, {stack.max()}]")
    
    
    if channel >= stack.shape[2]:
        print(f"warning: {channel}not exist")
        channel = 0
    
    data = stack[:, :, channel, :, :]
    print(f"Processed data shape: {data.shape} (time, Z, Y, X)")
    
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
# 2. Complete 3D cell area detection and displacement calculation
# ----------------------------
def detect_cell_region_3d(volume, top_percentile=95, min_size_ratio=0.05):
    
    depth, height, width = volume.shape
    center_ratio = 0.3
    crop_d = int(depth )
    crop_h = int(height * center_ratio)
    crop_w = int(width * center_ratio)
    center_x = width // 2 - crop_w // 2
    center_y = height // 2 - crop_h // 2
    center_z = depth // 2 - crop_d // 2
    return (center_x, center_y, center_z, crop_w, crop_h, crop_d)

def extract_cell_region_3d(volume, bbox):
  
    x, y, z, w, h, d = bbox
    return volume[z:z+d, y:y+h, x:x+w]

def compute_3d_displacement_complete(vol1, vol2, bbox):
    """Complete 3D displacement field calculation"""

    cell_vol1 = extract_cell_region_3d(vol1, bbox)
    cell_vol2 = extract_cell_region_3d(vol2, bbox)
    
    
    # Calculate 2D optical flow for each z layer
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
    
    
    for z_idx in range(1, cell_vol1.shape[0]-1):
        # Estimate the z-direction displacement using the displacement changes of adjacent layers
        disp_above = np.sqrt(displacement_3d_cell[z_idx-1, :, :, 0]**2 + displacement_3d_cell[z_idx-1, :, :, 1]**2)
        disp_below = np.sqrt(displacement_3d_cell[z_idx+1, :, :, 0]**2 + displacement_3d_cell[z_idx+1, :, :, 1]**2)
        displacement_3d_cell[z_idx, :, :, 2] = 0.5 * (disp_above + disp_below) * 0.05
    
    # Create a displacement field of the overall volume size, but with values only in the cell region.
    x, y, z, w, h, d = bbox
    full_displacement_3d = np.zeros((vol1.shape[0], vol1.shape[1], vol1.shape[2], 3), dtype=np.float32)
    full_displacement_3d[z:z+d, y:y+h, x:x+w] = displacement_3d_cell
    
    
    return full_displacement_3d, displacement_3d_cell, (cell_vol1, cell_vol2)

def calculate_3d_strain_complete(displacement_3d_cell):
    """Complete 3D strain field calculation"""
    # Enlarge the displacement field to facilitate better calculation of the gradient
    displacement_scaled = displacement_3d_cell * 100.0
    
    # Calculate 3D gradient
    gradients = []
    for i in range(3):  # For each displacement component
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
# 3. Stress Calculation Module
# ----------------------------
def calculate_3d_stress_cell_region(strain_3d):
    """Calculate the 3D stress field based on strain analysis"""
    
    
    # material parameter
    E = 28.0  
    nu = 0.25  
    
   
    mu = E / (2 * (1 + nu))
    lambda_ = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))  # 拉梅第一常数
   
    d, h, w = strain_3d[0, 0].shape
    
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
   
    # Calculate the normal stress components
    stress_3d[0, 0] = lambda_ * volumetric_strain + 2 * mu * epsilon_xx  # σ_xx
    stress_3d[1, 1] = lambda_ * volumetric_strain + 2 * mu * epsilon_yy  # σ_yy
    stress_3d[2, 2] = lambda_ * volumetric_strain + 2 * mu * epsilon_zz  # σ_zz
    
    # Calculate the shear stress components
    stress_3d[0, 1] = 2 * mu * epsilon_xy  # σ_xy
    stress_3d[1, 0] = 2 * mu * epsilon_xy  # σ_yx (对称)
    
    stress_3d[0, 2] = 2 * mu * epsilon_xz  # σ_xz
    stress_3d[2, 0] = 2 * mu * epsilon_xz  # σ_zx (对称)
    
    stress_3d[1, 2] = 2 * mu * epsilon_yz  # σ_yz
    stress_3d[2, 1] = 2 * mu * epsilon_yz  # σ_zy (对称)
    
    sigma_xx_mean = stress_3d[0, 0].mean()
    sigma_yy_mean = stress_3d[1, 1].mean()
    sigma_zz_mean = stress_3d[2, 2].mean()
    
    # Calculate the hydrostatic pressure
    hydrostatic_pressure = -(sigma_xx_mean + sigma_yy_mean + sigma_zz_mean) / 3.0
    print(f"  Average hydrostatic pressure = {hydrostatic_pressure:.8f} Pa")
    
    return stress_3d, mu

def calculate_traction_stress_cell(stress_3d, normal_vectors=None):
    
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

def calculate_principal_stresses_cell(stress_3d):
    """Calculate the principal stresses and principal directions"""
    
    d, h, w = stress_3d[0, 0].shape
    
    # Initialize the principal stress and principal direction
    principal_stresses = np.zeros((d, h, w, 3), dtype=np.float32)  # 三个主应力
    principal_directions = np.zeros((d, h, w, 3, 3), dtype=np.float32)  # 三个主方向
    
    # Calculate the eigenvalues and eigenvectors of the stress tensor for each voxel
    for i in range(d):
        for j in range(h):
            for k in range(w):
                # Construct the stress tensor
                stress_tensor = np.array([
                    [stress_3d[0, 0, i, j, k], stress_3d[0, 1, i, j, k], stress_3d[0, 2, i, j, k]],
                    [stress_3d[0, 1, i, j, k], stress_3d[1, 1, i, j, k], stress_3d[1, 2, i, j, k]],
                    [stress_3d[0, 2, i, j, k], stress_3d[1, 2, i, j, k], stress_3d[2, 2, i, j, k]]
                ])
                
                # Calculate eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(stress_tensor)
                
                # Sort by the size of the eigenvalues
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                principal_stresses[i, j, k] = eigenvalues
                principal_directions[i, j, k] = eigenvectors.T
   
    return principal_stresses, principal_directions

# ----------------------------
# 4. Advanced 3D Visualization Function
# ----------------------------
def create_simple_3d_visualization_with_stress(displacement, strain_volume, stress_components=None):
    """Create a simplified 3D visualization"""
    z, y, x = displacement.shape[:3]
    
    # Create a simple scatter plot to display the points with larger displacements
    disp_magnitude = np.sqrt(displacement[...,0]**2 + displacement[...,1]**2 + displacement[...,2]**2)
    
    # Locate the points with significant displacement
    threshold = disp_magnitude.mean() + disp_magnitude.std()
    significant_points = disp_magnitude > threshold
    
    Z_idx, Y_idx, X_idx = np.where(significant_points)
    
    # Create displacement visualization
    
    fig_disp = go.Figure(data=go.Scatter3d(
        x=X_idx, y=Y_idx, z=Z_idx,
        mode='markers',
        marker=dict(
            size=3,
            color=disp_magnitude[Z_idx, Y_idx, X_idx],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="displacement amplitude")
        )
    ))
    
    fig_disp.update_layout(
        title='3D Displacement Distribution (Significant Displacement Points)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )


    strain_volume1 = strain_volume
    
    # Create strain visualization
    strain_threshold = strain_volume1.mean() + strain_volume1.std()
    significant_strain_mask = strain_volume1 > strain_threshold

# Select significant points
    
    Z_S_idx, Y_S_idx, X_S_idx=np.where(significant_strain_mask)
    fig_strain = go.Figure(data=go.Scatter3d(
        x=X_S_idx, y=Y_S_idx, z=Z_S_idx,
        mode='markers',
        marker=dict(
            size=3,
            color=strain_volume[Z_S_idx, Y_S_idx, X_S_idx],
            colorscale='Hot',
            opacity=0.8,
            colorbar=dict(title="strain value")
        )
    ))
    
    fig_strain.update_layout(
        title='3D strain distribution',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    
    # If there are stress data, create a stress visualization.
    if stress_components is not None:
        # Visualization of tensile stress amplitude
        traction_magnitude = stress_components['traction_magnitude']
        
        # Locate the points with higher tensile stress
        stress_threshold = traction_magnitude.mean() + traction_magnitude.std()
        significant_stress_points = traction_magnitude > stress_threshold
        Z_stress_idx, Y_stress_idx, X_stress_idx = np.where(significant_stress_points)
        
        fig_traction = go.Figure(data=go.Scatter3d(
            x=X_stress_idx, y=Y_stress_idx, z=Z_stress_idx,
            mode='markers',
            marker=dict(
                size=3,
                color=traction_magnitude[Z_stress_idx, Y_stress_idx, X_stress_idx],
                colorscale='Jet',
                opacity=0.8,
                colorbar=dict(title="Traction stress (Pa)")
            )
        ))
        
        fig_traction.update_layout(
            title='3D Traction Stress Distribution |τ| (Pa)',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        
        # Visualization of the maximum principal stress
        principal_stresses = stress_components['principal_stresses']
        max_principal_stress = principal_stresses[:, :, :, 0]  # 最大主应力
        
        # Locate the point with the greater principal stress
        principal_threshold = max_principal_stress.mean() + max_principal_stress.std()
        significant_principal_points = max_principal_stress > principal_threshold
        Z_principal_idx, Y_principal_idx, X_principal_idx = np.where(significant_principal_points)
        
        fig_principal = go.Figure(data=go.Scatter3d(
            x=X_principal_idx, y=Y_principal_idx, z=Z_principal_idx,
            mode='markers',
            marker=dict(
                size=3,
                color=max_principal_stress[Z_principal_idx, Y_principal_idx, X_principal_idx],
                colorscale='Reds',
                opacity=0.8,
                colorbar=dict(title="Maximum principal stress (Pa)")
            )
        ))
        
        fig_principal.update_layout(
            title='3D maximum principal stress distribution (Pa)',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        
        # Von Mises equivalent stress visualization
        s = stress_components['stress_tensor']
        von_mises = np.sqrt(0.5 * (
            (s[0,0] - s[1,1])**2 + 
            (s[1,1] - s[2,2])**2 + 
            (s[2,2] - s[0,0])**2 +
            6 * (s[0,1]**2 + s[0,2]**2 + s[1,2]**2)
        ))
        
        # Locate the points with relatively high equivalent stress
        vm_threshold = von_mises.mean() + von_mises.std()
        significant_vm_points = von_mises > vm_threshold
        Z_vm_idx, Y_vm_idx, X_vm_idx = np.where(significant_vm_points)
        
        fig_von_mises = go.Figure(data=go.Scatter3d(
            x=X_vm_idx, y=Y_vm_idx, z=Z_vm_idx,
            mode='markers',
            marker=dict(
                size=3,
                color=von_mises[Z_vm_idx, Y_vm_idx, X_vm_idx],
                colorscale='Hot',
                opacity=0.8,
                colorbar=dict(title="von Mises stress (Pa)")
            )
        ))
        
        fig_von_mises.update_layout(
            title='3D von Mises equivalent stress distribution (Pa)',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        
        return fig_disp, fig_strain, fig_traction, fig_principal, fig_von_mises
    
    return fig_disp, fig_strain

def visualize_stress_results_cell(vol1, stress_components, z_slice=None):
    """Two-dimensional visualization of stress results in the cell region"""
    d, h, w = vol1.shape
    
    if z_slice is None:
        z_slice = d // 2  # Default display of the middle layer
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # original image
    axes[0,0].imshow(vol1[z_slice], cmap='gray')
    axes[0,0].set_title(f'original image - Layer Z{z_slice}')
    
    # Traction stress amplitude
    traction_mag = stress_components['traction_magnitude']
    im1 = axes[0,1].imshow(traction_mag[z_slice], cmap='jet')
    axes[0,1].set_title('Traction stress amplitude |τ| (Pa)')
    plt.colorbar(im1, ax=axes[0,1])
    
    # maximum principal stress
    principal_max = stress_components['principal_stresses'][:, :, :, 0]
    im2 = axes[0,2].imshow(principal_max[z_slice], cmap='Reds')
    axes[0,2].set_title('maximum principal stress (Pa)')
    plt.colorbar(im2, ax=axes[0,2])
    
    # component of stress

    stress_names = ['σ_xx', 'σ_yy', 'σ_zz']
    stress_data = [
        stress_components['stress_tensor'][0, 0, z_slice],  # σ_xx
        stress_components['stress_tensor'][1, 1, z_slice],  # σ_yy
        stress_components['stress_tensor'][2, 2, z_slice]   # σ_zz
    ]
    
    for i, (data, name) in enumerate(zip(stress_data, stress_names)):
        im = axes[1,i].imshow(data, cmap='viridis')
        axes[1,i].set_title(f'{name} (Pa)')
        plt.colorbar(im, ax=axes[1,i])
    
    plt.tight_layout()
    plt.show()

def visualize_2d_results(vol1, vol2, bbox, displacement, strains, z_slice=None):
    """Visualization of 2D results"""
    x, y, z, w, h, d = bbox
    
    if z_slice is None:
        z_slice = d // 2  # Select the middle layer display
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image comparison
    axes[0,0].imshow(vol1[z + z_slice], cmap='gray')
    axes[0,0].set_title(f'Time Point 1 - Z layer{z_slice}')
    axes[0,0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
    
    axes[0,1].imshow(vol2[z + z_slice], cmap='gray')
    axes[0,1].set_title(f'Time Point 2 - Z layer{z_slice}')
    axes[0,1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
    
    # displacement amplitude
    disp_magnitude = np.sqrt(displacement[:, :, :, 0]**2 + 
                            displacement[:, :, :, 1]**2 + 
                            displacement[:, :, :, 2]**2)
    im1 = axes[0,2].imshow(disp_magnitude[z_slice], cmap='hot')
    axes[0,2].set_title('displacement amplitude')
    plt.colorbar(im1, ax=axes[0,2])
    
    # displacement vector
    y_idx, x_idx = np.mgrid[0:h:10, 0:w:10]
    u = displacement[z_slice, ::10, ::10, 0]
    v = displacement[z_slice, ::10, ::10, 1]
    axes[0,3].quiver(x_idx, y_idx, u, v, scale=0.1, scale_units='xy', angles='xy')
    axes[0,3].set_title('XY plane displacement vector')
    axes[0,3].set_aspect('equal')
    
    # components of strain
    strain_names = ['ε_xx', 'ε_yy', 'ε_zz', 'equivalent strain']
    strain_data = [
        strains[0,0][z_slice],  # epsilon_xx
        strains[1,1][z_slice],  # epsilon_yy
        strains[2,2][z_slice],  # epsilon_zz
        np.sqrt(0.5 * ((strains[0,0][z_slice] - strains[1,1][z_slice])**2 + 
                      (strains[1,1][z_slice] - strains[2,2][z_slice])**2 + 
                      (strains[2,2][z_slice] - strains[0,0][z_slice])**2))
    ]
    
    for i, (data, name) in enumerate(zip(strain_data, strain_names)):
        im = axes[1,i].imshow(data, cmap='viridis')
        axes[1,i].set_title(name)
        plt.colorbar(im, ax=axes[1,i])
    
    plt.tight_layout()
    plt.show()

def create_elegant_2d_visualization(vol1, vol2, bbox, displacement, stress_components, z_slice=None):
    """Create an attractive 2D visualization: the left displacement overlay graph, the right von Mises stress graph"""
    
    x, y, z, w, h, d = bbox
    
    if z_slice is None:
        z_slice = d // 2  # Select the middle layer display
    
    # Create professional-looking graphics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subgraph: Displacement superposition diagram
    # Create an RGB image: Red represents time point 1, and green represents time point 2.
    cell_vol1_slice = vol1[z + z_slice, y:y+h, x:x+w]
    cell_vol2_slice = vol2[z + z_slice, y:y+h, x:x+w]
    
    # Normalized image
    img1_norm = (cell_vol1_slice - cell_vol1_slice.min()) / (cell_vol1_slice.max() - cell_vol1_slice.min())
    img2_norm = (cell_vol2_slice - cell_vol2_slice.min()) / (cell_vol2_slice.max() - cell_vol2_slice.min())
    
    # Create an RGB overlay image
    rgb_image = np.zeros((h, w, 3))
    rgb_image[..., 0] = img1_norm  
    rgb_image[..., 1] = img2_norm  
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Display superimposed image
    im1 = ax1.imshow(rgb_image, aspect='equal')
    ax1.set_title('Cell displacement overlay diagram\n(Red: Time Point 1, Green: Time Point 2)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('X pixels', fontsize=12)
    ax1.set_ylabel('Y pixels', fontsize=12)
    
    # Add the displacement vector field (sparse display)
    spacing = 8  # Vector spacing
    y_vec, x_vec = np.mgrid[0:h:spacing, 0:w:spacing]
    u_vec = displacement[z_slice, ::spacing, ::spacing, 0]
    v_vec = displacement[z_slice, ::spacing, ::spacing, 1]
    
    #Calculate the displacement amplitude for color mapping
    disp_magnitude_slice = np.sqrt(displacement[z_slice, :, :, 0]**2 + 
                                  displacement[z_slice, :, :, 1]**2)
    disp_magnitude_vec = disp_magnitude_slice[::spacing, ::spacing]
    
    # Draw the displacement vector
    quiver = ax1.quiver(x_vec, y_vec, u_vec, v_vec, disp_magnitude_vec,
                       scale=0.3, scale_units='xy', angles='xy', 
                       cmap='plasma', width=0.003, alpha=0.8)
    
    # Add displacement vector color bar
    cbar1 = plt.colorbar(quiver, ax=ax1, shrink=0.8)
    cbar1.set_label('displacement amplitude', fontsize=11)
    cbar1.ax.tick_params(labelsize=10)
    
    # Right subgraph: von Mises stress distribution
    # Obtain the stress tensor and calculate the von Mises equivalent stress
    s = stress_components['stress_tensor']  
    sxx = s[0, 0, z_slice, :, :]  # σxx
    syy = s[1, 1, z_slice, :, :]  # σyy
    szz = s[2, 2, z_slice, :, :]  # σzz
    
    # Extract the shear stress components (non-diagonal elements)
    sxy = s[0, 1, z_slice, :, :]  # τxy
    sxz = s[0, 2, z_slice, :, :]  # τxz
    syz = s[1, 2, z_slice, :, :]  # τyz
    
    von_mises = np.sqrt(0.5 * (
        (sxx - syy)**2 + 
        (syy - szz)**2 + 
        (szz - sxx)**2 +
        6 * (sxy**2 + sxz**2 + syz**2)
    ))

    # Use professional color mapping to display von Mises stress
    im2 = ax2.imshow(von_mises, cmap='inferno', aspect='equal')
    ax2.set_title('von Mises equivalent stress distribution', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('X pixel', fontsize=12)
    ax2.set_ylabel('Y pixel', fontsize=12)
    
    # Add stress contour lines
    if np.any(von_mises > 0):
        contour_levels = np.linspace(von_mises.min(), von_mises.max(), 8)
        contours = ax2.contour(von_mises, levels=contour_levels, 
                              colors='white', linewidths=0.5, alpha=0.6)
        ax2.clabel(contours, inline=True, fontsize=8, fmt='%.2e')
    
    # Add stress color bar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('von Mises stress (Pa)', fontsize=11)
    cbar2.ax.tick_params(labelsize=10)
   
    plt.tight_layout()
  
    fig.suptitle('Cell Mechanics Analysis - 2D Slice Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(False)
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    try:
        # 1. data loading
        file_path = "D:/转移文件/吴旭东老师的求助/video/Experiment-1424.tif"
        data = load_data(file_path, channel=1)
        
        # 2. Select the time frame to be compared
        time1, time2 =0,1
        vol1 = data[time1]
        vol2 = data[time2]
        
        
        # 3. Extract the central area
        center_ratio = 0.4
        vol1_center, bbox_center = extract_center_region_3d(vol1, center_ratio)
        vol2_center, _ = extract_center_region_3d(vol2, center_ratio)
        
        # 4. Detect the cell area (using the complete method)
        bbox = detect_cell_region_3d(vol1_center)
        
        # 5. Calculate 3D displacement (using the complete method)
        displacement_full, displacement_cell, cell_volumes = compute_3d_displacement_complete(
            vol1_center, vol2_center, bbox)
        
        # 6. Calculate 3D strain (using the complete method)
        strain_3d = calculate_3d_strain_complete(displacement_cell)
        
        # 7. Calculating stress
       
        stress_3d, mu = calculate_3d_stress_cell_region(strain_3d)
        
        # 8.Calculate the traction stress
        traction_stress, traction_magnitude = calculate_traction_stress_cell(stress_3d)
        
        # 9. Calculate the principal stresses
        principal_stresses, principal_directions = calculate_principal_stresses_cell(stress_3d)
        
        # 10. Build the stress component dictionary
        stress_components = {
            'stress_tensor': stress_3d,
            'traction_magnitude': traction_magnitude,
            'principal_stresses': principal_stresses,
            'traction_vectors': traction_stress
        }
        
        
        visualize_2d_results(vol1_center, vol2_center, bbox, displacement_cell, strain_3d, z_slice=displacement_cell.shape[0]//2)
        visualize_stress_results_cell(cell_volumes[0], stress_components, z_slice=displacement_cell.shape[0]//2)
        
       
        # Extract the strain components
        strains_components = [
            strain_3d[0,0],  # ε_xx
            strain_3d[1,1],  # ε_yy  
            strain_3d[2,2],  # ε_zz
            strain_3d[0,1],  # ε_xy
            strain_3d[0,2],  # ε_xz
            strain_3d[1,2]   # ε_yz
        ]
        
        # Calculate equivalent strain
        equivalent_strain = np.sqrt(0.5 * (
            (strains_components[0] - strains_components[1])**2 + 
            (strains_components[1] - strains_components[2])**2 + 
            (strains_components[2] - strains_components[0])**2 +
            6 * (strains_components[3]**2 + strains_components[4]**2 + strains_components[5]**2)
        ))
        create_elegant_2d_visualization(vol1_center, vol2_center, bbox, displacement_cell, stress_components, z_slice=displacement_cell.shape[0]//2)
        # Simplified 3D visualization (including stress)
        
        fig_disp_simple, fig_strain_simple, fig_traction_simple, fig_principal_simple, fig_von_mises_simple = create_simple_3d_visualization_with_stress(
            displacement_cell, equivalent_strain, stress_components
        )

        fig_disp_simple.show()
        fig_strain_simple.show()
        fig_traction_simple.show()
        fig_principal_simple.show()
        fig_von_mises_simple.show()
        
    except Exception as e:
        print(f"find bug: {e}")
        import traceback
        traceback.print_exc()
