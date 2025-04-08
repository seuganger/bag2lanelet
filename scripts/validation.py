import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

sys.setrecursionlimit(100000)

def load_osm_file(osm_file):   
    tree = ET.parse(osm_file)
    root = tree.getroot()
    for relation in root.findall(".//relation"):
        for member in relation.findall("member"):
            role = member.get("role")
            if role == "left":
                left_id = member.get("ref")
            elif role == "right":
                right_id = member.get("ref")

    nodes = {}
    for node in root.findall(".//node"):
            node_id = node.get("id")
            tags = {tag.get("k"): tag.get("v") for tag in node.findall("tag")}        
            x = float(tags.get("local_x"))
            y = float(tags.get("local_y"))
            z = float(tags.get("ele"))
            nodes[node_id] = (x, y, z)

    # print(nodes)
    for way in root.findall(".//way"):
        if way.get("id") == left_id:
            left_nodes = []
            for nd in way.findall("nd"):
                ref = nd.get("ref")
                if ref in nodes:
                    left_nodes.append(nodes[ref])
        elif way.get("id") == right_id:
            right_nodes = []
            for nd in way.findall("nd"):
                ref = nd.get("ref")
                if ref in nodes:
                    right_nodes.append(nodes[ref])

    # print(left_nodes,right_nodes)
    left_nodes = np.asarray(left_nodes)
    right_nodes = np.asarray(right_nodes)

    return left_nodes,right_nodes

def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')
    return ca[i, j]

def frdist(p, q):
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)
    len_p = len(p)
    len_q = len(q)
    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')    
    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)
    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist

def interp_line(line):
# interpolate
    t = np.linspace(0, len(line)-1, num = len(line))
    t_new = np.linspace(0, len(line)-1, num = (len(line)*50))
    fx = interp1d(t,line[:,0],kind='cubic')
    fy = interp1d(t,line[:,1],kind='cubic')
    fz = interp1d(t,line[:,2],kind='cubic')
    nline = np.asarray([fx(t_new),fy(t_new),fz(t_new)]).T
    return nline

def dis_line2gt(line, gt):
    st = 0
    end = len(gt)
    dis = np.zeros(len(line))
    this_dis = np.zeros(len(gt))
    for index in range(len(line)):
        print(index,st,end)
        for i in range(st,end):
            this_dis[i] = np.linalg.norm(gt[i] - line[index])
        dis[index] = np.min(this_dis[st:end])
        st = max(np.argmin(this_dis[st:end]) + st - 200, 0)
        end = min(np.argmin(this_dis[st:end]) + st + 200, len(gt))
    return dis

osm_file = "scripts/line2lanelet.osm"
gt = "scripts/ground truth.osm"
left_nodes,right_nodes = load_osm_file(osm_file)
left_gt,right_gt = load_osm_file(gt)
left_gt = interp_line(left_gt)
right_gt = interp_line(right_gt)
dis = dis_line2gt(right_nodes, right_gt)
print(np.mean(dis),np.max(dis),np.min(dis))
plt.hist(dis, bins=10, alpha=0.75)
plt.show()
