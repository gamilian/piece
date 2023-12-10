import numpy as np
import tensorflow as tf
import JIgsawAbitraryNetROI
import Parameters
import os, glob
import TFRecordIOWithROI
import cv2
import sys
import Utils
import PairwiseAlignment2Image

def SingleTest(checkpoint_root, K, net, is_training=False):
    input = tf.compat.v1.placeholder(tf.float32, [None, net.params['height'], net.params['width'], net.params['depth']])
    roi_box = tf.compat.v1.placeholder(tf.float32, [None, 4])

    logits = net._inference(input, roi_box, is_training)
    probability = tf.nn.softmax(logits)

    '''restore sessions'''
    sessions = []
    saver = tf.compat.v1.train.Saver(max_to_keep=2)
    for i in range(K):
        check_point = os.path.join(checkpoint_root, "g%d" % i)
        sess = tf.compat.v1.Session()
        sess_init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(sess_init_op)
        saver.restore(sess, tf.train.latest_checkpoint(check_point + '/'))
        print("restore model %d...Done!" % i)
        sessions.append(sess)

    while not net.close:
        if len(np.shape(net.evaluate_image)) < 4:
            net.evaluate_image = np.reshape(net.evaluate_image, [1, net.params['height'], net.params['width'], net.params['depth']])
        if len(np.shape(net.roi_box)) < 2:
            net.roi_box = np.reshape(net.roi_box, [1, 4])

        preds = []
        probs = []  # correct and incorrect probability
        for i in range(K):
            pred, prob = sessions[i].run([net.pred, probability], feed_dict={input: net.evaluate_image, roi_box: net.roi_box})
            pred = pred[0]
            prob = prob[0]
            preds.append(pred)
            probs.append(prob)
        yield preds, probs

    '''close'''
    for sess in sessions:
        sess.close()

def meassure_pairwise(alignments, fragments_dir, net, evaluator, K, Alpha, bg_color, save_all_leaner=False):
    f1 = open(os.path.join(fragments_dir, "filtered_alignments.txt"), 'w+')

    for alignment in alignments.data:
        v1 = alignment.frame1
        v2 = alignment.frame2
        rank = alignment.rank
        trans = alignment.transform
        raw_stitch_line = alignment.stitchLine

        # neural network judgement
        image1 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v1 + 1)))
        image2 = cv2.imread(os.path.join(fragments_dir, "fragment_{0:04}.png".format(v2 + 1)))
        item = PairwiseAlignment2Image.FusionImage2(image1, image2, trans, bg_color)
        if len(item) <= 0:
            continue
        path_img, overlap_ratio, transform_offset = item[0], item[1], item[2]
        resized_path_img = cv2.resize(path_img, (Parameters.NNHyperparameters["height"], Parameters.NNHyperparameters["width"]))

        net.evaluate_image = resized_path_img
        [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio] = Utils.ConvertRawStitchLine2BBoxRatio(raw_stitch_line, path_img, trans, transform_offset, max_expand_threshold=32)
        net.roi_box = [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]
        preds, probs = next(evaluator)
        for i in range(K):
            if preds[i]==1:
                preds[i]=1
            else:
                preds[i]=-1

        '''ensemble result'''
        correct_probs = []
        for i in range(len(probs)):
            correct_probs.append(probs[i][1])
        correct_probability = np.average(correct_probs)
    
        sign = np.sum(np.multiply(preds, Alpha))
        # if sign > 0:
        #     final_class = 1
        # else:
        #     final_class = 0
        if correct_probability>0.5:
            final_class = 1
        else:
            final_class = 0
        
        if correct_probability > 0.4:
            cv2.imwrite(os.path.join(fragments_dir, "test" , f"fusion_{v1 + 1}_{v2 + 1}_{trans[0][0]}{trans[0][1]}_{correct_probability}.png"), item[0])
            f1.write("%d\t%d\t%f\t0\n" % (v1, v2, correct_probability))
            f1.write(correct_probs.__str__())   
            f1.write("%f %f %f\n%f %f %f\n0 0 1\n" % (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]))
    
    f1.close()

    print("meanssure_pairwise complete!")


def main(_):
    K = Parameters.NNHyperparameters["learner_num"]                    # the number of learner for boost training
    params = Parameters.NNHyperparameters
    checkpoint_root = Parameters.WorkSpacePath['checkpoint_dir']
    
    '''evaluation '''
    measure_data_root1 = Parameters.WorkSpacePath["example_measure_root"]
    fragments_dirs = glob.glob(os.path.join(measure_data_root1, "*_ex"))

    with open(os.path.join(checkpoint_root, "alpha.txt")) as f:
        for line in f:
            line = line.rstrip()
            if line[0] != '#':
                line = line.split()
                Alpha = [float(x) for x in line]

    net = JIgsawAbitraryNetROI.JigsawNetWithROI(params=Parameters.NNHyperparameters)
    evaluator = SingleTest(checkpoint_root=checkpoint_root, K=5, net=net, is_training=False)

    for i in range(len(fragments_dirs)):
        print(f"dataset {i+1}/{len(fragments_dirs)}:  {fragments_dirs[i]}")
        if not os.path.exists(os.path.join(fragments_dirs[i], "alignments.txt")):
            continue
        bg_color_file = os.path.join(fragments_dirs[i], "bg_color.txt")
        with open(bg_color_file) as f:
            for line in f:
                line = line.split()
                if line:
                    bg_color = [int(i) for i in line]
                    bg_color = bg_color[::-1]
    
        relative_alignment = os.path.join(fragments_dirs[i], "alignments.txt")
        alignments = Utils.Alignment2d(relative_alignment)
        meassure_pairwise(alignments, fragments_dirs[i], net, evaluator, K, Alpha, bg_color, save_all_leaner=False)
        print("----------------")



if __name__ == "__main__":


    tf.compat.v1.app.run()
