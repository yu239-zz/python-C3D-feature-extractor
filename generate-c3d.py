#!/usr/bin/python
# generate-c3d.py>>

import cv2
import sys
import os
import sys
import glob
import struct
import numpy as np
import cPickle as pickle

__sample_frames__ = 16    # by default
__c3d_length__ = 4096     # fc6 length
__c3d_root__ = os.path.expanduser("~")+"/C3D/examples/c3d_feature_extraction"
__batch_size__ = 20       
__force_computing__ = False
__video_width__ = 400.0   # resize the video

def preprocessing(videos_list, gpu):
    with open(videos_list, "r") as f:
        files = f.read().splitlines()

    finput = open("/tmp/c3d-input-list-%d.txt" % gpu, "w")
    foutput = open("/tmp/c3d-output-list-%d.txt" % gpu, "w")

    total = 0
    for v in files:
        print v

        video_dir = os.path.splitext(v)[0]
        c3d_dir = video_dir + "/c3d"
        video_name = os.path.split(video_dir)[1]
        ## skip already computed
        if (not __force_computing__) and os.path.isfile(video_dir+"/"+video_name+".c3d"):
            continue

        os.system("mkdir -p " + c3d_dir)

        cap = cv2.VideoCapture(v)
        frames_n = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        scale = __video_width__ / width
        new_height = int(scale * height)
        new_width = int(scale * width)

        if len(glob.glob("%s/*.jpg" % c3d_dir)) != frames_n:
            print "===============", frames_n, "==============="
            cnt = 1
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.resize(frame, (new_width,new_height))
                cv2.imwrite("%s/%06d.jpg" % (c3d_dir,cnt), frame)
                cnt += 1

        start_frame = 1
        while start_frame-1+__sample_frames__ <= frames_n:
            total += 1
            finput.write("%s/ %d 0\n" % (c3d_dir, start_frame))
            foutput.write("%s/%06d\n" % (video_dir, start_frame))
            start_frame += __sample_frames__

        cap.release()

    finput.close()
    foutput.close()
    return total

def call_c3d_script(total, gpu):
    num_batches = (total+__batch_size__-1) / __batch_size__
    abs_input_path = "/tmp/c3d-input-list-%d.txt" % gpu
    abs_output_path = "/tmp/c3d-output-list-%d.txt" % gpu
    # first replace the input list in the prototxt file
    with open("%s/prototxt/c3d_sport1m_feature_extractor_frm.prototxt" % __c3d_root__, "r") as f:
        lines = f.read().splitlines()
    update_lines = ['source: "%s"' % abs_input_path if "source" in line else line for line in lines]
    update_lines = ['batch_size: %d' % __batch_size__ if "batch_size" in line else line for line in update_lines]
    with open("%s/prototxt/feature_extractor_frm-%d.prototxt" % (__c3d_root__,gpu), "w") as f:
        for line in update_lines:
            f.write(line + "\n")
    # then rewrite the script to include the output list 
    with open("%s/c3d_sport1m_feature_extraction_frm.sh" % __c3d_root__, "r") as f:
        c3d_cmd_args = f.read().split()
    
    c3d_cmd_args[2] = "prototxt/feature_extractor_frm-%d.prototxt" % gpu
    c3d_cmd_args[4] = str(gpu)
    c3d_cmd_args[5] = str(__batch_size__)
    c3d_cmd_args[6] = str(num_batches)
    c3d_cmd_args[7] = abs_output_path
    new_c3d_script = "feature_extraction_frm-%d.sh" % gpu
    os.system('cd %s; echo "%s" > %s' % (__c3d_root__, " ".join(c3d_cmd_args), new_c3d_script))
    # call
    ret = os.system('cd %s; sh %s 2>&1 | grep -v "h264"' % (__c3d_root__, new_c3d_script))
    ## shouldn't proceed if this call fails
    assert ret == 0


def process_c3d_features(videos_list):
    def read_binary_fc6(fc6):
        fin = open(fc6, "rb")
        s = fin.read(20)
        length = struct.unpack("i", s[4:8])[0]
        assert length == __c3d_length__, "c3d feature length error"
        feature = fin.read(4*length)
        feature = struct.unpack("f"*length, feature)
        return list(feature)
    
    with open(videos_list, "r") as f:
        files = f.read().splitlines()
    for v in files:
        print "collecting c3d features:", v
        video_dir = os.path.splitext(v)[0]
        video_name = os.path.split(video_dir)[1]
        ## skip already computed
        if (not __force_computing__) and os.path.isfile(video_dir+"/"+video_name+".c3d"):
            continue

        fc6s = sorted(glob.glob("%s/*.fc6-1" % video_dir))
        feats_movie = []
        for fc6 in fc6s:
            tmp_feat = read_binary_fc6(fc6)
            feats_movie.append(tmp_feat)
        ## handle the case when the clip is too short (< __sample_frames__)
        ## just write a c3d feature with all 0s
        if not feats_movie:
            feats_movie.append([0]*__c3d_length__)
            print >> sys.stderr, "warning: %s is too short; dummy c3d features are used" % v

        with open(video_dir+"/"+video_name+".c3d", "w") as f:
            pickle.dump(np.array(feats_movie,dtype=np.float32), f, pickle.HIGHEST_PROTOCOL)

        os.system("rm -f %s/*.fc6-1" % video_dir)
        ## remove jpg files
        os.system("rm -rf %s/c3d/" % video_dir)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "usage: ./generate-c3d.py <videos-list> <gpu-id>"
        sys.exit()
        
    gpu = int(sys.argv[2])

    total = preprocessing(sys.argv[1], gpu)

    call_c3d_script(total, gpu)
        
    process_c3d_features(sys.argv[1])

    print "generating c3d features done."
    
# <<generate-c3d.py
