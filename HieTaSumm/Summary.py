from .Files import Files
from .Frame import Frame
from .Models import Models
from .Graph import Graph
import os 
from PIL import Image
import numpy as np
from scipy import spatial
import networkx as nx
import cv2 as cv

class Summary:
  def __init__(self, dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method):
    self.delta_t = rate * time
    self.video_file = "{}/{}/".format(dataset_frames, video)
    self.video = video
    self.alpha = alpha
    self.percent = percent
    self.graph = Graph(is_binary, hierarchy)
    self.model = Models(selected_model)
    self.frame = Frame(self.model)
    self.len_shot = 0
    self.rate = rate

    # creating path for files
    self.input_graph_file = Files('{}graph.txt'.format(self.video_file))
    self.input_mst = '{}mst_{}_{}.txt'.format(self.video_file, self.percent, self.alpha)
    self.input_higra = Files('{}higra_{}_{}.txt'.format(self.video_file, self.percent, self.alpha))
    self.cut_graph_file = Files('{}cut_graph_{}_{}.txt'.format(self.video_file, self.percent, self.alpha))
    self.frames_path = "{}frames/".format(self.video_file)
    self.output_skim = '{}skim_{}_{}'.format(self.video_file, self.percent, self.alpha)
    self.summ_path = '{}{}/'.format(self.video_file, gen_summary_method['method'])
    self.summ_input = Files('{}{}.txt'.format(self.video_file, gen_summary_method['method']))

    cut_number = self.bestCutNumber()

    print("----------------------")
    print("Processing video {}".format(self.video_file))

    if(not os.path.exists(self.input_graph_file.file)):
      f = open(self.input_graph_file.file, "a") # pensando em paralelizar para garantir a integridade do arquivo
      f.close()
      features_list = self.model.features(self.frames_path) # extract features
      self.frame.load(self.frames_path, self.delta_t, self.input_graph_file, features_list) # Load the frame list and create a graph for the video

    if(not os.path.exists(self.input_higra.file)):

      if gen_summary_method['method'] == 'n_fixed_keyframes':
        chosen = gen_summary_method['n_keyframes']
        self.cut_number = chosen - 1
      elif gen_summary_method['method'] in ['group_central_frames', 'percent_spaced']:
        self.len_shot = round((len(os.listdir(self.frames_path))) * (self.percent/100)) - 1
        self.cut_number = int(self.bestCutNumber() * alpha)
        if(self.cut_number <= 2):
            self.cut_number = 3

      #tree = gen_mst(input_graph_file, input_mst) # generate the minimum spanning tree
      tree = self.input_graph_file.read_graph_file(Files(), cut_graph = False, cut_number = 0) # read the graph file
      leaflist = self.graph.compute_hierarchy(tree, self.input_higra) # Create the hierarchy based on the minimum spanning tree and return the leaves of the new hierarchy
      cuted_graph = self.graph.cut_graph(self.input_higra, self.cut_graph_file, cutNumber = cut_number) # Create a new graph based on the hierarchy and the level cut
      #plotGraph(cuted_graph, True)
      if gen_summary_method['method'] == 'sequential_keyframe':
        self.sequential_keyframe(gen_summary_method['n_keyframes'])
      if gen_summary_method['method'] in ('n_fixed_keyframes', 'percent_spaced'):
        self.get_n_frames(cuted_graph, leaflist)
      if gen_summary_method['method'] == 'group_central_frames':
        self.group_central_frames(cuted_graph, leaflist)

      if(not os.path.exists(self.output_skim)):
        os.mkdir(self.output_skim)
      self.generate_video()


  def bestCutNumber(self):
    if(os.path.exists(self.frames_path)):
        frame_list = os.listdir(self.frames_path)
        frame_list.sort() # to garanted the time order
        features_list = []
        for frames in frame_list:
            if frames.endswith("jpg"):
                frame_dir = self.frames_path + frames
                features_list.append(self.rgbSim(frame_dir))
        # features_list = [rgbSim(video_file + frames) for frames in frame_list]
        weight_list = []
        feat_list_len = len(features_list)

        for vertex1 in range(feat_list_len):
            for vertex2 in range(self.calc_init(vertex1, self.delta_t, feat_list_len),
                                    self.calc_end(vertex1, self.delta_t, feat_list_len)):
                w = self.spatialSim(features_list[vertex1], features_list[vertex2])/100 # teste nan
                weight_list.append(w)
        cut = np.std(weight_list)
        while cut<=1:
            cut = cut * 10
        return(np.round(cut) - 1)

  def rgbSim(self, frame_dir):
      frame = Image.open(frame_dir)
      frame_reshape = frame.resize((round(frame.size[0]*0.5), round(frame.size[1]*0.5)))
      frame_array = np.array(frame_reshape)
      frame_array = frame_array.flatten()
      frame_array = frame_array/255
      return frame_array

  def spatialSim(self, frame1, frame2):
      similarity = 100 * (-1 * (spatial.distance.cosine(frame1, frame2) - 1))
      if similarity < 20:
          similarity = 20
      return similarity
  
  def sequential_keyframe(self, n):
      for i in range(1, n+1):
        kf = str(i).zfill(6)
        if not os.path.isdir(self.summ_path):
          os.mkdir(self.summ_path)
        os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
        self.summ_input.save_graph_data(kf, '  ', '.jpg')

  def get_n_frames(self, graph, leaflist):
    S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    #KF_list = []
    for c in range(len(S)):
      central_node = len(S[c].nodes)
      comp_leaf_list = []
      for i in range(central_node):
        if(list(S[c])[i] in leaflist):
          comp_leaf_list.append(list(S[c])[i])
      cn = int(len(comp_leaf_list)/2)
      if not (cn == 0):
        kf = str(comp_leaf_list[cn]).zfill(6)

        if not os.path.isdir(self.summ_path):
          os.mkdir(self.summ_path)
        os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
        self.summ_input.save_graph_data(kf, '  ', '.jpg')

  def group_central_frames(self, graph, leaflist):
    S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    #KF_list = []
    self.len_shot = int(self.len_shot / len(S))
    for c in range(len(S)):
      central_node = len(S[c].nodes)
      comp_leaf_list = []
      for i in range(central_node):
        if(list(S[c])[i] in leaflist):
          comp_leaf_list.append(list(S[c])[i])
      len_leaf_list = len(comp_leaf_list)
      cn = int(len_leaf_list/2) # find the central node for keyframe strategy

      if not (cn == 0):
        if(len_leaf_list < self.len_shot):
            init_keyshots = 0
            #if(len_shot == 1):
              #end_keyshots = 1
            #else:
            end_keyshots = len_leaf_list - 1
        else:
            init_keyshots = cn - (int(self.len_shot/2))
            end_keyshots = cn + (int(self.len_shot/2))

        if not os.path.isdir(self.summ_path):
          os.mkdir(self.summ_path)

        for k in range(init_keyshots, end_keyshots):
            keyshot = str(comp_leaf_list[k]).zfill(6) # save on keyshot the central node
            os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, keyshot, self.summ_path, keyshot))
            self.summ_input.save_graph_data(keyshot, '  ', '.jpg') # (path for keyshot, each frame of keyshot, validator to save, extension)

    # Video Generating function
  def generate_video(self):
    image_folder = self.summ_input[:-1]#'.' # make sure to use your folder

    images = [img for img in os.listdir(image_folder)

            if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if any

    frame = cv.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape
    os.chdir(self.output_skim)

    video = cv.VideoWriter(self.video + '.mp4', 0, self.rate, (width, height))

    # Appending the images to the video one by one
    for image in images:
      video.write(cv.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv.destroyAllWindows()
    video.release() # releasing the video generated

  def calc_init(self, i, delta_t, frame_len):
      if((i < delta_t) or delta_t < 0):
          return 0
      elif((i + delta_t) > frame_len):
          return i
      else:
          return i - delta_t

  def calc_end(self, i, delta_t, frame_len):
      if(((i + delta_t) > frame_len) or delta_t < 0):
          return frame_len
      else:
          return i + delta_t