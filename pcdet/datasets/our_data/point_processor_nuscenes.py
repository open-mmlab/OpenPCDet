import numpy as np
from .base_point_processor import BasePointProcessor, RCS_MEAN, RCS_STD, use_SNR
#Mean RCS: 8.70, Std RCS: 7.04

# New logs: Overall RCS mean across frames: -6.38, Overall RCS std across frames: 8.14
NUSCENE_RCS_MEAN = 8.70
NUSCENE_RCS_STD = 7.04

Z_OFFSET = 0.9

class PointProcessorNuscenes(BasePointProcessor):
    def __init__(self, radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames, is_rcs_normalized=False, COMP_POINTS_MOTION = False):
        super().__init__(radar_offset_tx, radar_offset_ty, radar_offset_yaw, n_frames, COMP_POINTS_MOTION)
        self.is_rcs_normalized = is_rcs_normalized

    

    def rotate_points(self, points, shift_x, shift_y, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 1. Rotate FIRST
        x_rotated = points[:, 0] * cos_yaw - points[:, 1] * sin_yaw
        y_rotated = points[:, 0] * sin_yaw + points[:, 1] * cos_yaw

        # 2. Add translation SECOND
        points[:, 0] = x_rotated - shift_x
        points[:, 1] = y_rotated + shift_y

        points = self.rotate_velocities(points, yaw)

        return points

    def rotate_velocities(self, points, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        vx = points[:, 4].copy()
        vy = points[:, 5].copy()
        
        points[:, 4] = vx * cos_yaw - vy * sin_yaw
        points[:, 5] = vx * sin_yaw + vy * cos_yaw

        return points


    def calculate_compensated_velocity(self, points, shift_x, shift_y, shift_yaw, vx, vy, v_yaw, timestamp_pc):
        """
        Calculates the absolute compensated radial velocity for radar points.
        
        Args:
            points: (N, 7) numpy array where columns are [x, y, z, intensity, RCS, max_v_r, v_r, v_r_comp, time]
        
        Returns:
            v_comp: (N,) numpy array of compensated velocities
        """
        v_comp = super().calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, vx, vy, v_yaw, timestamp_pc)
        v_comp_x = v_comp * self.u_x
        v_comp_y = v_comp * self.u_y

        v_comp_list = np.column_stack([v_comp_x, v_comp_y])


        return v_comp_list
    
    
    def processPointsSingleFrame(self, points, timestamp_pc, shift_x=0.0, shift_y=0.0, shift_yaw=0.0, vx=None, vy=None, v_yaw=None):
        #### WATCH OUT: OFFSET ON Z DUE TO RADAR PLACEMENT, CHECK IF IT MATCHES THE ONE IN THE DATASET
        #points[:, 2] -= Z_OFFSET  # Adjust Z for radar mounting height  
        # points = points[np.logical_and(points[:, 2] >= -3, points[:, 2] <= 5)]
        #points[:, 2] = 0  # Set Z to 0 for BEV processing

        v_comp = self.calculate_compensated_velocity(points, shift_x, shift_y, shift_yaw, vx, vy, v_yaw, timestamp_pc)

        #print("Speed: ", np.shape(radial_ambiguous_velocity))
        #v_comp=np.expand_dims(v_comp, axis=1)
        if use_SNR:
            snr = np.expand_dims(points[:,4], axis=1)
        else: #RCS
            snr = points[:,3]
            if self.is_rcs_normalized:
                snr = self.convert_intensity_to_rcs(snr)
                
            # ####
            # self.bins.append(np.histogram(snr, range=(-30, 70), bins=100)[0])
            # rcs_mean = np.mean(snr)
            # rcs_std = np.std(snr)

            # self.means.append(rcs_mean)
            # self.stds.append(rcs_std)
            # ###

            snr = np.expand_dims(snr, axis=1)

        time_vector = np.zeros((points.shape[0], 1), dtype=points.dtype)
        processed_points = np.hstack([points[:, 0:3], snr, v_comp, time_vector])
        
        processed_points = self.filter_invalid_points(processed_points)
        # [x, y, z, snr, v_comp_x, v_comp_y, time]
        
        # print("Processed points shape: ", np.shape(processed_points))

        return processed_points
    
    def alignRCSDistribution(self, rcs):
        rcs = ((rcs - RCS_MEAN) / RCS_STD) * NUSCENE_RCS_STD + NUSCENE_RCS_MEAN
        print("Aligned RCS stats - mean: {:.2f}, std: {:.2f}".format(np.mean(rcs), np.std(rcs)))
        return rcs

    def updateTimestamp(self, timestamp):
        return timestamp + self.dt

    def filter_valid_speed_points(self, points):
        # Filter out points with unrealistic speeds (e.g., > 30 m/s)
        speed = np.sqrt(points[:, -2]**2 + points[:, -3]**2)  
        valid_speed_points = points[speed < 60]
        return valid_speed_points

    def compensate_points_motion(self, points, dt):
        points[:, 0] = points[:, 0] + points[:, 4] * dt
        points[:, 1] = points[:, 1] + points[:, 5] * dt

        return points
