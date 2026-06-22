import numpy as np
from abc import abstractmethod

use_SNR = False
ALIGN_RCS_DISTRIBUTION = False

RCS_MEAN = -5.23
RCS_STD = 15.30

class BasePointProcessor:
    def __init__(self, radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames, COMP_POINTS_MOTION = False):
        self.radar_offset_tx = radar_offset_tx
        self.radar_offset_ty = radar_offset_ty
        self.radar_offset_yaw = np.deg2rad(radar_offset_yaw)
        self.COMP_POINTS_MOTION = COMP_POINTS_MOTION

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_yaw = 0.0
        self.img = None

        self.bins = []
        self.means = []
        self.stds = []
        self.new_pc_arrived = False

        self.timestamp_last_frame_left = 0
        self.timestamp_last_frame_right = 0

        self.timestamp_current_odom = 0
        self.timestamp_last_odom = 0

        self.previous_vel_x = 0.0
        self.previous_vel_y = 0.0
        self.previous_vel_yaw = 0.0

        self.n_frames = n_frames
        self.points_per_frame = []
        self.timestamp_last_frame = 0
        self.dt = 0
        self.multiframe_points = np.empty((0, 7), dtype=np.float32)  # [x, y, z, intensity, time, frame_id, velocity]

    def add_timestamp(self, timestamp):
        if self.timestamp_last_frame != 0:
            self.dt = (timestamp - self.timestamp_last_frame) * 1e-9  # Convert nanoseconds to seconds

        #print(f"New frame timestamp: {timestamp}, dt from last frame: {self.dt:.3f} seconds")
        
        self.timestamp_last_frame = timestamp

    def add_auxiliar_cloud(self, points, timestamp, shift_x = 0.0, shift_y = 0.0, yaw = 0.0, v_x = None, v_y = None, v_yaw = None):
        # print("**** DEBUG *********")
        # print(f"Point 0 : {points[0]}, shift x, y, yaw {shift_x, shift_y, yaw}")
        yaw_in_radians = np.deg2rad(yaw)
        # print(f"Yaw in radians {yaw_in_radians}")

        total_shift_x = shift_x + self.radar_offset_tx
        total_shift_y = shift_y + self.radar_offset_ty
        total_shift_yaw = yaw_in_radians + self.radar_offset_yaw

        processed_points = self.processPointsSingleFrame(points, timestamp, total_shift_x, total_shift_y, total_shift_yaw, v_x, v_y, v_yaw)
        # print(f"Point 0 after processing: {processed_points[0]}")

        #Adding time factor that shift the points of the auxiliary cloud compared to the main cloud
        dt_aux = (timestamp - self.timestamp_last_frame) * 1e-9  # Convert nanoseconds to seconds
        
        if v_x is not None and v_y is not None and v_yaw is not None:
            current_v_x, current_v_y, current_v_yaw = v_x, v_y, v_yaw
        else:
            current_v_x, current_v_y, current_v_yaw = self.calculate_interpolated_velocity(timestamp)

        additional_shift_x = dt_aux * (current_v_x - current_v_yaw * total_shift_y)  
        additional_shift_y = dt_aux * (current_v_y + current_v_yaw * total_shift_x) 
        additional_shift_yaw = current_v_yaw * dt_aux

        # print(f"Auxiliary cloud timestamp: {timestamp}, main cloud timestamp: {self.timestamp_last_frame}, dt_aux: {dt_aux:.3f} seconds")
        # print(f"Current velocity (x, y, yaw): {current_v_x:.2f} m/s, {current_v_y:.2f} m/s, {np.rad2deg(current_v_yaw):.2f} deg/s")
        # print(f"Additional shift for auxiliary cloud due to velocity: {additional_shift_x:.2f} m, {additional_shift_y:.2f} m, {np.rad2deg(additional_shift_yaw):.2f} deg")

        shift_x -= additional_shift_x
        shift_y += additional_shift_y
        yaw_in_radians += additional_shift_yaw

        # print(f"Total shift applied to auxiliary cloud: {shift_x:.2f} m, {shift_y:.2f} m, {np.rad2deg(yaw_in_radians):.2f} deg")

        rotated_points = self.rotate_points(processed_points, shift_x, shift_y, yaw_in_radians)
        # print(f"Point 0 after rotation: {rotated_points[0]}")

        n_points_before = self.multiframe_points.shape[0]
        self.points_per_frame[-1] += rotated_points.shape[0]

        self.multiframe_points = np.vstack([self.multiframe_points, rotated_points])

        #print(f"Added {rotated_points.shape[0]} points from auxiliary cloud. Total points before: {n_points_before}, after: {self.multiframe_points.shape[0]}")

        #print(f"Tot points in vector: {sum(self.points_per_frame)}")
        #print(f"Current multiframe points shape: {self.multiframe_points.shape}")

        return 

    
    def processPoints(self, points, vx = None, vy=None, v_yaw=None):
        self.new_pc_arrived = False

        if vx is None or vy is None or v_yaw is None:
            vx, vy, v_yaw = self.calculate_interpolated_velocity(self.timestamp_last_frame)

        processed_points = self.processPointsSingleFrame(points, self.timestamp_last_frame, self.radar_offset_tx, self.radar_offset_ty, self.radar_offset_yaw, vx, vy, v_yaw)


        if len(self.points_per_frame) >= self.n_frames:
            self.multiframe_points = self.multiframe_points[self.points_per_frame[0]:, :]
            self.points_per_frame.pop(0)  # Remove the oldest frame

        self.points_per_frame.append(len(processed_points))

        self.multiframe_points = self.transposeFrame(self.multiframe_points, vx, vy, v_yaw)
        self.multiframe_points[:, 6] = self.updateTimestamp(self.multiframe_points[:, 6])  

        if self.COMP_POINTS_MOTION:
            self.multiframe_points = self.compensate_points_motion(self.multiframe_points, self.dt)

        self.multiframe_points = np.vstack([self.multiframe_points, processed_points])


        # print(f"Tot points in vector: {sum(self.points_per_frame)}")
        # print(f"Current multiframe points shape: {self.multiframe_points.shape}")
        # if len(self.points_per_frame) == 5:
        #     print(f"Sample of points frame 5: {self.multiframe_points[0:2, :]}")
        #     print(f"Sample of points frame 4: {self.multiframe_points[self.points_per_frame[0]:self.points_per_frame[0]+2, :]}")
        #     print(f"Sample of points frame 3: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]:self.points_per_frame[0]+self.points_per_frame[1]+2, :]}")
        #     print(f"Sample of points frame 2: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+2, :]}")
        #     print(f"Sample of points frame 1: {self.multiframe_points[self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]:self.points_per_frame[0]+self.points_per_frame[1]+self.points_per_frame[2]+self.points_per_frame[3]+2, :]}")

        return self.multiframe_points

    def transposeFrame(self, points, vx, vy, v_yaw):
        d_theta = -v_yaw * self.dt
        cos_yaw = np.cos(d_theta)
        sin_yaw = np.sin(d_theta)

        tx = self.radar_offset_tx
        ty = self.radar_offset_ty

        # 1. Move points from the Old Radar bumper to the Old IMU center
        p_imu_x = points[:, 0] + tx
        p_imu_y = points[:, 1] + ty

        vx = vx - v_yaw * self.radar_offset_ty
        vy = vy + v_yaw * self.radar_offset_tx

        # 2. Translate the points backwards by the IMU's movement
        p_shifted_x = p_imu_x - vx * self.dt
        p_shifted_y = p_imu_y + vy * self.dt

        # 3. Rotate the points around the new IMU center (Left-Handed coordinates)
        p_rotated_x = p_shifted_x * cos_yaw - p_shifted_y * sin_yaw
        p_rotated_y = p_shifted_x * sin_yaw + p_shifted_y * cos_yaw

        # 4. Move the points from the New IMU center back out to the New Radar bumper
        points[:, 0] = p_rotated_x - tx
        points[:, 1] = p_rotated_y - ty

        # 5. Rotate velocities 
        # (Velocities are directional vectors, they do not suffer from Lever Arm translation)
        points = self.rotate_velocities(points, d_theta)

        return points

    def add_random_z(self,points):
    
        N = points.shape[0]
        
        # If the frame is completely empty, just return it
        if N == 0:
            return points

        # ==========================================
        # TRICK 1: THE Z-AXIS INFLATION
        # ==========================================
        
        points[:, 2] = np.random.uniform(0.3, 1.5, size=N)

        return points

    def snr_to_fake_rcs(self, points, snr_mean=None, snr_std=None):
        VOD_RCS_MEAN = -12.0
        VOD_RCS_STD = 12.0
        
        # Extract your raw SNR column
        snr = points[:, 3]
        
        # If not provided SNR stats, calculate them on the fly for this frame
        if snr_mean is None:
            snr_mean = np.mean(snr)
        if snr_std is None:
            snr_std = np.std(snr) + 1e-6 # Add tiny epsilon to prevent division by zero
            
        # Standardize SNR, then scale it to VoD's RCS distribution
        fake_rcs = ((snr - snr_mean) / snr_std) * VOD_RCS_STD + VOD_RCS_MEAN
        
        # Overwrite the SNR column with our fake RCS values
        points[:, 3] = fake_rcs
        
        return points

    def convert_intensity_to_rcs(self, rcs_norm):
        MAX_RCS = 100
        MIN_RCS = -100

        rcs = rcs_norm * (MAX_RCS - MIN_RCS) + MIN_RCS

        #Overall RCS mean across frames: -5.23, Overall RCS std across frames: 15.30
        

        # filtered_rcs = rcs[rcs != 0]  # Filter 
        # self.bins.append(np.histogram(filtered_rcs, range=(-60, 40), bins=1000)[0])

        # #debug
        
        # rcs_mean = np.mean(filtered_rcs)
        # rcs_std = np.std(filtered_rcs)

        # self.means.append(rcs_mean)
        # self.stds.append(rcs_std)

        if ALIGN_RCS_DISTRIBUTION:
            
            rcs = self.alignRCSDistribution(rcs)

            # print("RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(rcs), np.std(rcs)))
            # print("Sample RCS values: ", rcs[:10])

        return rcs

    def filter_invalid_points(self, points):
        #valid_points = points[(np.abs(points[:, 0]) > 1.0) & (np.abs(points[:, 0]) < 51.2) & (np.abs(points[:, 1]) > 1.0) & (np.abs(points[:, 1]) < 25.6)]  # Filter out points with x=0 (assuming these are invalid)
        valid_points = points[(np.abs(points[:, 0]) > 1.0) & (np.abs(points[:, 1]) > 1.0)]

        valid_points = self.filter_valid_speed_points(valid_points)

        return valid_points
    
    def print_bins(self):
        if self.bins:
            overall_histogram = np.sum(self.bins, axis=0)
            print(f"Overall RCS histogram (sum of all frames): {overall_histogram}")
        print(f"Overall RCS mean across frames: {np.mean(self.means):.2f}, Overall RCS std across frames: {np.mean(self.stds):.2f}")
        print(f"Average std of RCS across frames: {np.mean(self.stds):.2f}")

    def calculate_interpolated_velocity(self, timestamp_pc):
        if self.timestamp_last_odom == 0:
            return self.vel_x, self.vel_y, self.vel_yaw  # No previous velocity, return current as is

        interpolated_vel_x = self.interpolate1d(self.timestamp_last_odom, self.timestamp_current_odom, timestamp_pc, self.previous_vel_x, self.vel_x)
        interpolated_vel_y = self.interpolate1d(self.timestamp_last_odom, self.timestamp_current_odom, timestamp_pc, self.previous_vel_y, self.vel_y)
        interpolated_vel_yaw = self.interpolate1d(self.timestamp_last_odom, self.timestamp_current_odom, timestamp_pc, self.previous_vel_yaw, self.vel_yaw)

        return interpolated_vel_x, interpolated_vel_y, interpolated_vel_yaw
        
    def interpolate1d(self, x0, x1, xt, y0, y1):
        if x1 - x0 == 0:
            return y0  # Avoid division by zero, return y0 as fallback
        return y0 + (y1 - y0) * ((xt - x0) / (x1 - x0))
    
    def add_odometry(self, vel_x, vel_y, vel_yaw, timestamp):
        if self.timestamp_last_odom == 0: #First odometry message fill both previous and current
            self.previous_vel_x = vel_x
            self.previous_vel_y = vel_y
            self.previous_vel_yaw = vel_yaw
            self.timestamp_last_odom = timestamp
        else:
            self.previous_vel_x = self.vel_x
            self.previous_vel_y = self.vel_y
            self.previous_vel_yaw = self.vel_yaw
            self.timestamp_last_odom = self.timestamp_current_odom

        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_yaw = vel_yaw

        self.timestamp_current_odom = timestamp

    def calculate_compensated_velocity(self, points, shift_x, shift_y, shift_yaw, v_x, v_y, v_yaw, timestamp_pc):
        """
        Calculates the absolute compensated radial velocity for radar points.
        
        Args:
            points: (N, 7) numpy array where columns are [x, y, z, intensity, RCS, max_v_r, v_r, v_r_comp, time]
            timestamp_pc: Timestamp of the point cloud
        
        Returns:
            v_comp: (N,) numpy array of compensated velocities
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # print("points shape: ", points.shape)

        v_meas = points[:, 6] # The raw Doppler velocity from the radar
        
        if v_x is None or v_y is None or v_yaw is None:
            v_x, v_y, omega_z = self.calculate_interpolated_velocity(timestamp_pc)
            # print(f"Interpolated velocity used for compensation: v_x={v_x:.2f} m/s, v_y={v_y:.2f} m/s, v_yaw={omega_z:.2f} rad/s")
        else:
            v_x, v_y, omega_z = v_x, v_y, v_yaw
            # print(f"Interpolated velocity used for compensation from parameters: v_x={v_x:.2f} m/s, v_y={v_y:.2f} m/s, v_yaw={omega_z:.2f} rad/s")
        
        t_x, t_y, yaw = shift_x, shift_y, shift_yaw  
        
        # radar sensor's physical velocity
        v_sens_x = v_x - (omega_z * t_y)
        v_sens_y = v_y + (omega_z * t_x)
        
        # Rotate velocity into the radar's local coordinate frame
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        v_rad_x = v_sens_x * cos_y + v_sens_y * sin_y
        v_rad_y = -v_sens_x * sin_y + v_sens_y * cos_y
        
        # Distance from radar to point
        dist = np.sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero for points exactly at (0,0,0)
        dist = np.clip(dist, a_min=1e-6, a_max=None)
        
        self.u_x = x / dist
        self.u_y = y / dist
        
        # Ego velocity
        v_ego_los = (v_rad_x * self.u_x) + (v_rad_y * self.u_y)
        
        # Compensate the raw measurement
        v_comp = v_meas + v_ego_los

        # print("EGO VELOCITY (X, Y, YAW): ", v_x, v_y, omega_z)
        # print("MEAN RADIAL VELOCITY BEFORE COMPENSATION: ", np.mean(v_meas))
        # print("MEAN EGO LOS VELOCITY: ", np.mean(v_ego_los))
        # print("MEAN RADIAL VELOCITY AFTER COMPENSATION: ", np.mean(v_comp))
        
        return v_comp

    @abstractmethod
    def processPointsSingleFrame(self, points, timestamp_pc, shift_x, shift_y, shift_yaw, vx, vy, v_yaw):
        """Must be implemented by child classes (VoD vs NuScenes)"""
        pass
    
    @abstractmethod
    def rotate_points(self, points, shift_x, shift_y, yaw):
        """Must be implemented by child classes (VoD vs NuScenes)"""
        pass

    @abstractmethod
    def alignRCSDistribution(self, rcs):
        """Must be implemented by child classes (VoD vs NuScenes)"""
        pass

    @abstractmethod
    def updateTimestamp(self, timestamp):
        """Must be implemented by child classes (VoD vs NuScenes)"""
        pass

    @abstractmethod
    def filter_valid_speed_points(self, points):
        """Must be implemented by child classes (VoD vs NuScenes)"""
        pass

    @abstractmethod 
    def compensate_points_motion(self, points, dt):
        pass

    @abstractmethod
    def rotate_velocities(self, points, yaw):
        pass
