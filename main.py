from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input_videos/soccer.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

team_control_time = {team_assigner.team_colors[0]: 0, team_assigner.team_colors[1]: 0}

# Comptabiliser le temps de contrôle du ballon pour chaque équipe
for frame_num, team in enumerate(team_ball_control):
    if team == team_assigner.team_colors[0]:
        team_control_time[team_assigner.team_colors[0]] += 1
    elif team == team_assigner.team_colors[1]:
        team_control_time[team_assigner.team_colors[1]] += 1

# Dessiner les boîtes englobantes des joueurs et afficher l'équipe en contrôle du ballon
for frame_num, frame in enumerate(video_frames):
    for player_id, track in tracks['players'][frame_num].items():
        bbox = track['bbox']
        team_color = track['team_color']
        has_ball = track.get('has_ball', False)
        
        # Dessiner la boîte englobante du joueur
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), team_color, 2)
        
        # Si le joueur a le ballon, ajouter un indicateur
        if has_ball:
            cv2.putText(frame, 'Has Ball', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
    
    # Afficher l'équipe qui contrôle le ballon
    cv2.putText(frame, f'Team Control: {team_ball_control[frame_num]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Sauvegarder la vidéo avec les annotations
save_video('output_videos/annotated_soccer.mp4', video_frames)

  

if __name__ == '__main__':
    main()