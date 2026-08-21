# Pose-Estimation-To-Virtual-Environment




https://github.com/user-attachments/assets/e2391798-7063-4d4c-a3e3-9b3d1216d7ba



***

# Human Pose to PyBullet Avatar Mimic

Program tracks human body movements using a webcam and MediaPipe Pose, then maps the detected joints onto a 3D avatar in the PyBullet physics engine. 

To achieve realistic and smooth motion, the program utilizes several mathematical and geometric calculations, ranging from coordinate mapping to linear algebra for 3D rotations.

## Mathematical Calculations

### 1. Coordinate Mapping (2D Camera to 3D World)
MediaPipe outputs normalized coordinates $\mathbf{L} = (x, y, z)$ in the range of $[0, 1]$. The function `landmark_to_world` converts these into PyBullet 3D world coordinates $\mathbf{W} = (X, Y, Z)$:

$$ X = (x - 0.5) \times S $$
$$ Y = -z \times S \times 0.7 $$
$$ Z = (1.0 - y) \times S \times 0.7 $$

Where:
* $S$ is the `SCALE` factor (default 1.2) to adjust the overall size of the avatar.
* $x - 0.5$ shifts the horizontal center to the origin $(0,0,0)$.
* The $Z$-axis (height) is inverted because image $y$-coordinates increase downwards, while 3D $Z$-coordinates increase upwards.
* The factor $0.7$ is applied to dampen the depth and height ratios so the avatar's proportions look more natural.

### 2. Motion Smoothing (Exponential Moving Average)
To prevent the 3D avatar from jittering due to camera noise, the program applies an Exponential Moving Average (EMA) or Linear Interpolation (Lerp) to smooth the position vectors $\mathbf{P}$ over time:

$$ \mathbf{P'}_t = \mathbf{P}_{t-1} \times (1 - \alpha) + \mathbf{P}_t \times \alpha $$

Where:
* $\mathbf{P}_{t-1}$ is the smoothed position from the previous frame.
* $\mathbf{P}_t$ is the new target position from the current frame.
* $\alpha$ is the `SMOOTHING` factor (default 0.35). This means the avatar takes 35% of the new target position and retains 65% of the previous position, resulting in fluid motion.

### 3. Midpoint Calculation
To place the torso and head correctly, the program calculates the midpoint $\mathbf{M}$ between two corresponding joints (e.g., left and right shoulders $\mathbf{S}_L, \mathbf{S}_R$):

$$ \mathbf{M} = \frac{\mathbf{A} + \mathbf{B}}{2} $$

For example, the torso center is:
$$ \mathbf{M}_{torso} = \frac{\mathbf{S}_{left} + \mathbf{S}_{right}}{2} $$

**e. Quaternion Conversion:**
PyBullet uses Quaternions $\mathbf{q} = [q_x, q_y, q_z, q_w]$ for rotation to avoid gimbal lock. The Axis-Angle representation $(\hat{\mathbf{A}}, \theta)$ is converted to a Quaternion using the standard formula:

$$ \mathbf{q} = \left[ \hat{A}_x \sin\left(\frac{\theta}{2}\right), \quad \hat{A}_y \sin\left(\frac{\theta}{2}\right), \quad \hat{A}_z \sin\left(\frac{\theta}{2}\right), \quad \cos\left(\frac{\theta}{2}\right) \right] $$

The capsule is then placed at center $\mathbf{C}$ with rotation $\mathbf{q}$ using `p.resetBasePositionAndOrientation()`.
