import numpy as np
import json

def corn2xywh(boxes):
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    wh = boxes[..., 2:] - boxes[..., :2]
    # return tf.concat([xy,wh], axis=-1)
    return np.concatenate([xy,wh], axis=-1)

def xywh2corn(boxes):
    xymin = boxes[..., :2] - boxes[..., 2:] / 2.0
    xymax = boxes[..., :2] + boxes[..., 2:] / 2.0
    # return tf.concat([xymin, xymax], axis=-1)
    return np.concatenate([xymin, xymax], axis=-1)

def centroid_tracking(centroids_t0, centroids_t1): # shape[n, 2], shape[m, 2]
    t0_ids = [i for i in range(len(centroids_t0))]
    dist_2d = []
    for cen_t0 in centroids_t0:
        cen_t0 = cen_t0[None, :]
        dist_row = np.linalg.norm(cen_t0-centroids_t1, axis=-1)
        dist_2d.append(dist_row)
    
    dist_2d_row_wise_sort = sorted(zip(dist_2d, t0_ids), key=lambda x:min(x[0]))

    pairs = []
    selected_t1 = set()
    for dist_t0_to_t1, t0_id in dist_2d_row_wise_sort:
        t1_id = np.argmin(dist_t0_to_t1)
        if t1_id in selected_t1: continue
        else: selected_t1.add(t1_id)
        pairs.append((t0_id, t1_id))
        
    return pairs

def tracking_id(before_ids, pairs, after_boxes_len): # ex) before_ids = [100,101,102,103,104,105]
    current_id = max(before_ids)
    pairs = sorted(pairs, key=lambda x:x[1], reverse=True)
    after_ids = []
    for i in range(after_boxes_len):
        if pairs and i == pairs[-1][1]:
            idx0, _ = pairs.pop()
            after_ids.append(before_ids[idx0])
        else:
            current_id += 1
            after_ids.append(current_id)
    return after_ids
        
def light_shade(img, alpha=1.5, beta=1.0):
    img = (img - beta*128)*alpha + beta*128
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    
def json2label(path): # json 경로
    with open(path, "r") as f:
        data = json.load(f)
    labels = [shape["label"] for shape in data["shapes"]]
    boxes = [shape["points"] for shape in data["shapes"]]
    boxes = np.array([box[0] + box[1] for box in boxes])
    boxes = corn2xywh(boxes)
    imgsize = (data["imageWidth"], data["imageHeight"])
    return labels, boxes, imgsize # rank1, rank2, rank1
  
def normalize_boxes(imgsize, boxes): # xy, 0~512 -> 0~1
    imgsize = np.concatenate([imgsize, imgsize])
    boxes /= imgsize
    return boxes # rank2
  
def make_text_label(ids, boxes, path): # 숫자, 0~1boxes, 파일이름
    assert len(ids) == len(boxes)
    with open(path, 'w', encoding="utf-8") as f:
        for id, box in zip(ids, boxes):
            f.write("{} {} {} {} {}\n".format(id, *box))