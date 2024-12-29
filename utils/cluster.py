import numpy as np

class Cluster:

    def visualize(self, pts:np.ndarray):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw([pcd])

    def cluster(self, pred:np.ndarray, offsets:np.ndarray, X):
        """
        :param pred: (n, n_classes)
        :param offsets: (n, 3)
        :return: pred: (n, n_classes)
        """
        from sklearn.cluster import DBSCAN, KMeans
        def split_clusters(X, clusters):
            from sklearn.decomposition import PCA
            new_clusters = []
            for c in clusters:
                x = X[c, :2]
                x = PCA(2).fit_transform(x)

                length = x.max(0)-x.min(0)
                if np.any(length>10):
                    # split
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
                    idx = np.where(kmeans.labels_ == 1)[0]
                    new_clusters.append(c[idx])
                    idx = np.where(kmeans.labels_ == 0)[0]
                    new_clusters.append(c[idx])
                else:
                    new_clusters.append(c)
            return new_clusters

        n = len(pred)
        labels = np.argmax(pred, 1)

        # thing_idx = np.where((np.sum(offsets**2, 1)**0.5>0.01) | (labels>0))[0]    # both pred and offsets
        thing_idx = np.where(np.sum(offsets**2, 1)**0.5>0.01)[0]    # both pred and offsets
        X = X+offsets
        # self.visualize(X)

        # 1. get clusters
        clustering = DBSCAN(eps=1.05, min_samples=30).fit(X[thing_idx])
        clusters = [thing_idx[clustering.labels_ == i] for i in range(np.max(clustering.labels_) + 1)]
        clusters = split_clusters(X, clusters)

        # 2. sort clusters: phi
        # 16, 15, ...9, 1, 2, ..., 8
        cluster_centers = np.asarray([X[c].mean(0) for c in clusters])
        phi = np.arctan2(cluster_centers[:, 1]-cluster_centers[:, 1].max(), cluster_centers[:, 0])
        clusters = np.asarray(clusters, dtype=object)[np.argsort(phi)].tolist()

        for c in clusters:
            l = np.max(labels[c])
            if l > 0 and l < 9:
                # this is an upper jaw, rotate
                clusters = clusters[::-1]
                break
            elif l >=9:
                break

        # 3. assign probs to each cluster
        probs = np.zeros((len(clusters), 17))   # (n_clusters, n_classes)
        for i, idx in enumerate(clusters):
            # probs[i] = np.bincount(labels[idx], minlength=probs.shape[1]).astype(float) / len(idx)
            probs[i] = np.bincount(labels[idx], minlength=probs.shape[1])

        # 4. assign label to cluster
        # 4.1) assign the label with maximum probability to the cluster
        label_assigned = np.zeros(17, dtype=bool)
        cluster_label = -np.ones(len(clusters), dtype=int)
        while np.sum(probs) > 1e-7:
            idx = np.argmax(probs)
            i_cluster, l = idx // probs.shape[1], idx%probs.shape[1]
            if l == 0:
                cluster_label[i_cluster] = l
                probs[i_cluster, :] = 0
            elif not label_assigned[l]:
                cluster_label[i_cluster] = l
                label_assigned[l] = True
                probs[i_cluster, :] = 0
            else:
                prob_ls = np.argsort(-probs[i_cluster])
                assigned = False
                for l in prob_ls:
                    if probs[i_cluster, l] <=1e-7:
                        break
                    if l> 0 and not label_assigned[l]:
                        cluster_label[i_cluster] = l
                        label_assigned[l] = True
                        probs[i_cluster, :] = 0
                        assigned = True
                        break
                if not assigned:
                    l = int(np.argmax(probs[i_cluster]))
                    if l==8 or l==16 or len(clusters[i_cluster])<100:
                        # assign 17
                        # _i = np.where(label_assigned==False)[0][-1]
                        # cluster_label[i_cluster] = _i
                        # label_assigned[_i] = True
                        probs[i_cluster, :] = 0
                    elif l < 8:
                        if np.any(label_assigned[l+1:9]==False):
                            # move distal
                            # search the label not assigned
                            l_not_assigned = np.where(label_assigned==False)[0][-1]
                            for _i in range(l+1, 9):
                                if not label_assigned[_i]:
                                    l_not_assigned = _i
                                    break
                            for _i in range(l_not_assigned-1, l-1, -1):
                                _loc = np.where(cluster_label==_i)
                                if _loc > i_cluster:
                                    cluster_label[_loc] = _i+1
                                    label_assigned[_i+1] = True
                                else:
                                    l = _i + 1

                            cluster_label[i_cluster] = l
                            label_assigned[l] = True
                            probs[i_cluster, :] = 0
                        elif l == 1:
                            if np.any(label_assigned[9:17]==False):
                                # reassign 1 or 9
                                _loc = np.where(cluster_label==1)[0]
                                if _loc < i_cluster:
                                    probs[_loc, 9] = probs[i_cluster, 1]
                                    cluster_label[_loc] = -1

                                    cluster_label[i_cluster] = 1
                                    label_assigned[1] = True
                                    probs[i_cluster, :] = 0
                                else:
                                    probs[i_cluster, 9] = probs[i_cluster, 1]
                                    probs[i_cluster, 1] = 0
                            else:
                                cluster_label[i_cluster] = -1
                                probs[i_cluster, :] = 0
                        elif np.any(label_assigned[1:l]==False):
                            # move mesial
                            l_not_assigned = np.where(label_assigned==False)[0][-1]
                            for _i in range(1, l):
                                if not label_assigned[_i]:
                                    l_not_assigned = _i
                                    break
                            for _i in range(l_not_assigned+1, l-1):
                                _loc = np.where(cluster_label==_i)
                                if _loc < i_cluster:
                                    cluster_label[_loc] = _i-1
                                    label_assigned[_i-1] = True
                                else:
                                    l = _i - 1

                            cluster_label[i_cluster] = l
                            label_assigned[l] = True
                            probs[i_cluster, :] = 0
                        else:
                            cluster_label[i_cluster] = -1
                            probs[i_cluster, :] = 0
                    else:
                        # l >=9
                        if np.any(label_assigned[l+1:17]==False):
                            # move distal
                            l_not_assigned = np.where(label_assigned==False)[0][-1]
                            for _i in range(l+1, 17):
                                if not label_assigned[_i]:
                                    l_not_assigned = _i
                                    break
                            for _i in range(l_not_assigned-1, l-1, -1):
                                _loc = np.where(cluster_label==_i)
                                if _loc < i_cluster:
                                    cluster_label[_loc] = _i+1
                                    label_assigned[_i+1] = True
                                else:
                                    l = _i + 1

                            cluster_label[i_cluster] = l
                            label_assigned[l] = True
                            probs[i_cluster, :] = 0
                        elif l==9:
                            if np.any(label_assigned[1:9]==False):
                                _loc = np.where(cluster_label==9)[0]
                                if _loc < i_cluster:
                                    probs[i_cluster, 1] = probs[i_cluster, 9]
                                    probs[i_cluster, 9] = 0
                                else:
                                    probs[_loc, 1] = probs[i_cluster, 9]
                                    cluster_label[_loc] = -1

                                    cluster_label[i_cluster] = 9
                                    label_assigned[9] = True
                                    probs[i_cluster, :] = 0
                            else:
                                cluster_label[i_cluster] = -1
                                probs[i_cluster, :] = 0
                        elif np.any(label_assigned[9:l]==False):
                            # move mesial
                            l_not_assigned = np.where(label_assigned == False)[0][-1]
                            for _i in range(9, l):
                                if not label_assigned[_i]:
                                    l_not_assigned = _i
                                    break
                            for _i in range(l_not_assigned + 1, l):
                                _loc = np.where(cluster_label == _i)
                                if _loc > i_cluster:
                                    cluster_label[_loc] = _i - 1
                                    label_assigned[_i - 1] = True
                                else:
                                    l = _i - 1

                            cluster_label[i_cluster] = l
                            label_assigned[l] = True
                            probs[i_cluster, :] = 0
                        else:
                            cluster_label[i_cluster] = -1
                            probs[i_cluster, :] = 0

        # 5. assign labels
        for i, c in enumerate(clusters):
            l = cluster_label[i]
            if l > 0:
                pred[c] += np.eye(17)[l]*2.
        # pred = pred + np.eye(17)[labels]*1.
        pred = pred/ np.sum(pred, 1, keepdims=True)
        return pred
