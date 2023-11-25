import numpy as np

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3), stitchLine=None):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform
        self.stitchLine = stitchLine

        # rank between frame1 and frame2
        # self.rank = -1

class Alignment2d:
    def __init__(self, relative_transform_filename):
        self.data = []
        # for example, {'0 1': [0,1,2]} means from 0--1 to find data[0,1,2]
        self.mapIdpair2Transform = {}
        # for example, {'0 1 1': 0} means from 0--1 and 1st to find data[0]
        self.mapIdpairRank2Transform = {}
        # for example, {0: [0,1,2,100]} means from 0 to find data[0,1,2,100] in which either 0-x or x-0
        self.to_frame_data = {}
        # for example, {0: [0,1,2,100]} means from 0 to find data[0,1,2,100] in which either 0-x or x-0
        self.frame_to_data = {}

        with open(relative_transform_filename) as f:
            all_line = [line.rstrip() for line in f]

            for line in all_line:
                data_str_list = line.split()
                v1, v2, score, m1, m2, m3, m4, m5, m6, m7, m8, m9 = [t(s) for t, s in zip((
                    int, int, float,
                    float, float, float,
                    float, float, float,
                    float, float, float),
                    data_str_list[0:12])]
                transform = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])

                self.data.append(Transform2d(v1, v2, score, transform))

        self.data = sorted(self.data, key=operator.attrgetter('score'), reverse=True)
        self.data = sorted(self.data, key=operator.attrgetter('frame2'))
        self.data = sorted(self.data, key=operator.attrgetter('frame1'))

        for i, item in enumerate(self.data):
            idpair = f'{item.frame1} {item.frame2}'
            if idpair in self.mapIdpair2Transform:
                self.mapIdpair2Transform[idpair].append(i)
            else:
                self.mapIdpair2Transform[idpair] = [i]
            if item.frame1 in self.frame_to_data:
                self.frame_to_data[item.frame1].append(i)
            else:
                self.frame_to_data[item.frame1] = [i]
            if item.frame2 in self.to_frame_data:
                self.to_frame_data[item.frame2].append(i)
            else:
                self.to_frame_data[item.frame2] = [i]

        # for key, value in self.mapIdpair2Transform.items():
        #     for rank, index in enumerate(value):
        #         new_key = "%s %d" % (key, rank + 1)
        #         self.mapIdpairRank2Transform[new_key] = index
        #         self.data[index].rank = rank + 1




def calculate_euclidean_distance(point1, point2):
    '''
    Calculate the Euclidean distance between two points
    '''
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_color_similarity(point_color, current_color):
    '''
    Calculate color similarity
    '''
    # Assume colors are represented by RGB values ​​ranging from 0 to 255
    r1, g1, b1 = point_color
    r2, g2, b2 = current_color
    r1, g1, b1 = int(np.uint16(r1)), int(np.uint16(g1)), int(np.uint16(b1))
    r2, g2, b2 = int(np.uint16(r2)), int(np.uint16(g2)), int(np.uint16(b2))
    
    # Calculate the Euclidean distance of a color
    distance = np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

    # Calculate similarity based on distance
    similarity = 1 - (distance / (255 * (3 ** 0.5)))
    return similarity

def calculate_average_color(image, segments_sik):
    average_color = [0, 0, 0]
    for point in segments_sik:
        current_color = image[point[1]][point[0]]
        average_color[0] += int(np.uint16(current_color[0]))
        average_color[1] += int(np.uint16(current_color[1]))
        average_color[2] += int(np.uint16(current_color[2]))
    # print(len(segments_sik))
    average_color = [c / len(segments_sik) for c in average_color]
    return average_color


def transform_cloud(cloud, transform):
    '''
    Apply a rigid transformation to a point cloud(2d)
    '''
    # Convenient matrix matmul
    cloud_homogeneous = np.hstack((cloud, np.ones((len(cloud), 1))))
    transformed_cloud = (transform@(cloud_homogeneous.T)).T
    return transformed_cloud[:, :2]

def tranform_ponit(point, transform):
    '''
    Apply a rigid transformation to a point
    '''
    return (transform@(np.array([point[0], point[1], 1]).reshape(3, 1)))[0:2].reshape(2, )

def calculate_normals(points):
    '''
    Calculate normal vector
    '''    
    tangents = np.diff(points, axis=0)
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T

    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals

def longest_common_subsequence(image1, image2, cloud1, cloud2, initial_transform, threshold=2):
    """
    Find the longest common subsequence of two sequences, and allowing a certain error
    """
    m, n = len(cloud1), len(cloud2)
    dp = np.zeros((m + 1, n + 1))

    # Populate the dynamic programming matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # t = compute_color_similarity(image1[cloud1[i - 1][1]][cloud1[i - 1][0]],
            #                             image2[cloud2[j - 1][1]][cloud2[j - 1][0]])
            t = calculate_euclidean_distance(cloud1[i - 1], tranform_ponit(cloud2[j - 1], initial_transform))
            if t <= threshold:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the longest common subsequence
    lcs1, lcs2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        # if compute_color_similarity(image1[cloud1[i - 1][1]][cloud1[i - 1][0]],
        #                                 image2[cloud2[j - 1][1]][cloud2[j - 1][0]]) >= threshold:
        t = calculate_euclidean_distance(cloud1[i - 1], tranform_ponit(cloud2[j - 1], initial_transform))
        if t <= threshold:
            lcs1.append(i - 1)
            lcs2.append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs1)), list(reversed(lcs2))