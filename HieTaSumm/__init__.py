from .Summary import Summary
import os
import json
import cv2 as cv
import numpy as np
from datetime import timedelta
from pathlib import Path

class HieTaSumm:
    def __init__(self, **kwargs):
        self.f_score_epochs = []
        self.cusa_epochs = []
        self.cuse_epochs = []
        self.cov_epochs = []
        PACKAGEDIR = Path(__file__).parent.absolute()
        my_file = PACKAGEDIR / 'options.json'

        json_file = open(my_file, 'r')
        default_data = json.loads(json_file.read())
        dataset_videos = kwargs.get('video_path')
        if not dataset_videos:
            dataset_videos = default_data['video_path']
        dataset_frames = kwargs.get('summary_path')
        if not dataset_frames:
            dataset_frames = default_data['summary_path']
        percent = kwargs.get('percent')
        if not percent:
            percent = default_data['percent']
        percent = int(percent)
        alpha = kwargs.get('alpha')
        if not alpha:
            alpha = default_data['alpha']
        alpha = int(alpha)
        rate = kwargs.get('rate')
        if not rate:
            rate = default_data['rate']
        rate = int(rate)
        time = kwargs.get('time')
        if not time:
            time = default_data['time']
        time = int(time)
        hierarchy = kwargs.get('hierarchy')
        if not hierarchy:
            hierarchy = default_data['hierarchy']
        selected_model = kwargs.get('selected_model')
        if not selected_model:
            selected_model = default_data['selected_model']
        is_binary = kwargs.get('is_binary')
        if not is_binary:
            is_binary = default_data['is_binary']
        gen_summary_method = kwargs.get('gen_summary_method')
        if not gen_summary_method:
            gen_summary_method = default_data['gen_summary_method']
        
        if not os.path.isdir(dataset_frames):
            os.mkdir(dataset_frames)

        if(os.path.exists(dataset_videos)):
            video_list = os.listdir(dataset_videos)
            video_list.sort() # to guarantee order
            for i in video_list:
                print("------------------------")
                print("{}/{}/".format(dataset_videos, i))
                self.frame_extractor("{}/{}".format(dataset_videos, i), rate, dataset_frames)
        else:
            print(f"do not exist directory or file with path {dataset_videos}")

        if(os.path.exists(dataset_frames)):
            video_list = os.listdir(dataset_frames)
            video_list.sort() # to guarantee order
            for video in video_list:
                if video != '.ipynb_checkpoints':
                    fscore, cusa, cuse, cov = self.hierarchical_summarization(dataset_frames, video, rate, time, percent, alpha, gen_summary_method, is_binary, hierarchy, selected_model)
                    self.f_score_epochs.append(fscore)
                    self.cusa_epochs.append(cusa)
                    self.cuse_epochs.append(cuse)
                    self.cov_epochs.append(cov)
        print("----------------------")
        print(f'The avg fscore of all videos was: {np.mean(self.f_score_epochs)}')
        print(f'The avg CUSa of all videos was: {np.mean(self.cusa_epochs)}')
        print(f'The avg CUSe of all videos was: {np.mean(self.cuse_epochs)}')
        print(f'The avg Cov of all videos was: {np.mean(self.cov_epochs)}')



    def frame_extractor(self, video_file, rate, frames):
        # i.e if video of duration 30 seconds, saves 0.5 frame each second = 60 frames saved in total
        SAVING_FRAMES_PER_SECOND = 1/rate
        filename, _ = os.path.splitext(video_file)
        video = frames + "/" + filename.split('/')[-1]
        filename = video + "/frames/"
        if not os.path.isdir(video):
            os.mkdir(video)
        # make a folder by the name of the video file
        if not os.path.isdir(filename):
            os.mkdir(filename)
            # read the video file
            cap = cv.VideoCapture(video_file)
            fps = cap.get(cv.CAP_PROP_FPS)
            saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)

            saving_frames_durations = self.get_saving_frames_durations(cap, saving_frames_per_second)

            count = 0
            frame_number = 1
            while True:
                is_read, frame = cap.read()
                if not is_read:
                    # break out of the loop if there are no frames to read
                    break
                # get the duration by dividing the frame count by the FPS
                frame_duration = count / fps
                try:
                    # get the earliest duration to save
                    closest_duration = saving_frames_durations[0]
                except IndexError:
                    # the list is empty, all duration frames were saved
                    break
                if frame_duration >= closest_duration:
                    # if closest duration is less than or equals the frame duration,
                    # then save the frame
                    # frame_duration_formatted = formatTimeDelta(timedelta(seconds=frame_duration))
                    number = str(frame_number).zfill(6)
                    frame_number += 1
                    cv.imwrite(os.path.join(filename, "{}.jpg".format(number)), frame) # frame_duration_formatted)), frame)
                    # drop the duration spot from the list, since this duration spot is already saved
                    try:
                        saving_frames_durations.pop(0)
                    except IndexError:
                        pass
                # increment the frame count
                count += 1

    def get_saving_frames_durations(self, cap, saving_fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS))
        # use np.arange() to make floating-point steps

        for i in np.arange(0, clip_duration, saving_fps):
            s.append(i)
        return s

    def hierarchical_summarization(self, dataset_frames, video, rate, time, percent, alpha, gen_summary_method, is_binary, hierarchy, selected_model):
        summ = Summary(dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method)
        return summ.fscore, summ.mean_cusa, summ.mean_cuse, summ.cov_value

if __name__== "__main__":
    HieTaSumm()