import os
import cv2 
import numpy as np
import networkx as nx
from .Files import Files

class Frame:
  def __init__(self, model):
    self.model = model

  def load(self, video_file, delta_t, output_graph_file, features_list):
    if(os.path.exists(video_file)):
      RG = nx.Graph()
      frame_list = os.listdir(video_file)
      frame_list.sort() # to garanted the time order

      feat_list_len = len(features_list)
      #weight_list = []

      for vertex1 in range(feat_list_len):
        for vertex2 in range(vertex1,
                            self.calc_end(vertex1, delta_t, feat_list_len)):
          frame1 = os.path.join(video_file, frame_list[vertex1])
          frame2 = os.path.join(video_file, frame_list[vertex2])
          w = self.compute_similarity_img(frame1, frame2, features_list[vertex1], features_list[vertex2])
          RG.add_edge(frame1, frame2, weight = w)
          #weight_list.append(w)
          #output_graph_file.save_graph_data("{}, {}, {:.2f}".format(vertex1, vertex2, w) , ' ', ' ')
      return RG

  def compute_similarity_img(self, img_path_1, img_path_2, fea_vec_img1, fea_vec_img2): # img_path_1, img_path_2):
      filename1 = os.path.basename(img_path_1).split(".")[0]
      filename2 = os.path.basename(img_path_2).split(".")[0]

      # Compute cosine similarity
      sim_cos = self.calculate_similarity(fea_vec_img1, fea_vec_img2)

      # Read images
      im1 = cv2.resize(cv2.imread(img_path_1), dsize=self.model.img_size_model, interpolation = cv2.INTER_AREA)
      im2 = cv2.resize(cv2.imread(img_path_2), dsize=self.model.img_size_model, interpolation = cv2.INTER_AREA)

      # Concatenate images horizontally
      im12 = cv2.hconcat([im1, im2])

      # Save concatenated image
      folder = Files()
      dst_dir_cos_sim = "../report/cos_sim"
      folder.create_folder(dst_dir_cos_sim)
      dst_dir = f"{dst_dir_cos_sim}/{self.model.name}"
      folder.create_folder(dst_dir)

      new_filename = f"{filename1}_{filename2}"
      cv2.imwrite(f"{dst_dir}/{new_filename}.jpg", im12)

      return sim_cos

  def calculate_similarity(self, vector1, vector2):
      sim_cos = np.linalg.norm(vector1-vector2, 1) * 100 / len(vector2)
      return sim_cos

  def calc_end(self, i, delta_t, frame_len):
      if(((i + delta_t) > frame_len) or delta_t < 0):
          return frame_len
      else:
          return i + delta_t