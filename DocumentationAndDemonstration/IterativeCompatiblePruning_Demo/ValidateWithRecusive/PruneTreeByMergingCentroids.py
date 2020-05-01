"""
class PruneTreeByMergingCentroids:
    __init__:
        Parameters:
            prefixString: It is a predefined string which will be used when generating cluster ids.
            cltree: The base cltree on which the pruning mechanism will be applied.
                    This has to be an instance of CLTree.
            balancedPrune: It is a boolean flag, value of which indicates if we want to create a "balanced" clusters
                           in terms of number of data points each cluster get.
            useSilhouette: It is a boolean flag, value of which indicates if we want to use "silhouette" as our our
                           quality metric in the current instance or not.

        Operational Details:
                Suppose we have divided the whole customer space into 100 small groups by a CLTree and
                each group at most has 5 data-points (This means the underlying CLTree has 100 leaves and
                each leaf at most has 5 data points). If we look at the customer data space graphically after
                CLTree has performed, we should realize that the whole customer space is divided by hyper rectangles
                and each rectangle is bounded by the cut which has been made to create that rectangle
                (Classic decision tree and CLTree use the similar way to divide the whole data space gradually,
                 so if you search internet how decision tree divides the whole data space into numerous hyper
                 rectangles, you should be able to find a fair amount of quality material and same logic applies
                 here in the case of CLTree). Let’s say, we have a requirement that each cluster has to encompass
                at least 50 data points, otherwise it is too small to maintain. We have 100 small hyper-rectangles
                and each has at-most 5 data points each, and now we have a job at hand that we need to combine these
                100 hyper-rectangles gradually in a scientific way so that we can form groups of hyper-rectangles
                where each group will be big enough to encompass at least 50 data points.

                One of the better ways to represent a group of hyper-rectangles, when the goal is to measure how close
                they are from others in the space respectively, is to represent a rectangle by it’s centroid.

                What is “centroid”?
                Refer to here: https://en.wikipedia.org/wiki/Centroid

                How to calculate centroid of a rectangle?
                Refer to here:
                 https://www.engineeringintro.com/mechanics-of-structures/centre-of-gravity/centroid-of-rectangle/

                We have all the information needed to calculate centroids of all the hyper-rectangles.
                How?
                Well, all the leaves are hyper-rectangle.
                The boundary line of each hyper-rectangle at each dimension is represented by all the min and max
                of all the attributes of respectively of the underlying leaf.
                Now, all there is left to do is calculation the centroid for each hyper rectangle represented by the
                leaves of the CLTree.
                Once we get all the representation vectors (centroids of all the hyper-rectangle) we will compare
                the distance between one to other representation vectors and merge the pair together which are the
                closest by the distance metric. After the merge of a pair, we recalculate the representation vector
                of the pair by calculating weighted average of the two representation vector  and then replace the
                individuals by the pair in the active participation group and continue the same process until our
                condition is stratifying metric and quality metric is met.


    prune:
        Parameters:
            min_y: The "min_y" dictates the minimum number of data points required for a group to be
                   considered as a cluster. This is an instance level variable.
            searchMode: Boolean Flag, to indicate if the current pruning process, which is happending as part of some
                        grid search mechanism to calculate the best possible value for an hyper-parameter or not.
                        If the value of this flag is False that means, we have already decided what would the values
                        of all the relevant hyper parameters and therefore have freedom to modify the structure
                        of the underlying CLTree to optimize the search space.
                        If the value of the flag is True, we do not change the structure of the underlying CLTree, so
                        that we can reset all the modifications after calculating the quality metric at the end of the
                        process and revert back all changes made to the underlying CLTree and get back the original one,
                        which would be used for other values of the hyper parameter we are trying to optimize for.

        Operational Details: This is the driver, only external facing and wrapper function of the class. The pruning
                             process, whatever it may be, is controlled from here.
                Algorithm:
                    1. Validate and set the value of instance level variable "min_y".
                    2. Initialize instance level variable "maxSoFar"(will be used to create unique cluster id)
                       and "version".
                    3. Initialize the some active instance level variables, totalDataInstancesbyEachGroup,
                        finalRepresentationVectors, finalRepresentationVectorsToCluster, originalRepresentationVectors
                    4. Invoke, an internal function "initializeTouchedNodes" to set the "touching nodes" at each leaf
                       node level for the underlying CLTree.
                    5. Invoke, an internal function "initializeCentroids" to get the original set of hyper-rectangle
                        centroids by each leaf node. [If there are some "touching nodes", the combination
                        a leaf nodes and with all it's "touching nodes" appear once.].
                    6. Get the final "totalDataInstancesbyEachGroup", "finalRepresentationVectors",
                    "finalRepresentationVectorsToCluster" by invoking "pruningTreeByMergingRepresentationVectors" with
                       instance variable "originalRepresentationVectors" and "min_y".
                    7. Measure the final cluster statistics by invoking "clusterStatistics" function with
                       instance level "originalRepresentationVectors", "finalRepresentationVectors",
                       "finalRepresentationVectorsToCluster", "totalDataInstancesbyEachGroup" and store it in "result".
                    8. If we are not currently searching of an optimized value for an hyper parameter and had decided
                       on the values of all the parameters, then we are free to "prune" the underlying CLTree to
                       optimize the search space. The optimization of the search space for the cost of little
                       modification of the underlying CLTree doesn't change the result or any details other than
                       may be depth/height of some nodes where applicable. The process is performed by invoking
                       "pruningRedundantNodes" with the "root" of the underlying CLTree.
                    9. Return the final result to the calling function.

    identifyingTouchedRegions:
        Parameters:
            root: An instance of CLNode which is root of a CLTree instance.

        Operational Details:
            Definition:
                **This operation is only valid for a CLTree.**
                In a complex situation, it is possible that a broadly similar group of points got split.
                into several regions either because the group is bounded by an irregular shape or because
                in order to isolate one or more sparse region(s) from the original the underlying region,
                it is cut into more than one piece.

                A region, Y1, is said to touch another region, Y2, on the i-th dimension on the lower bounding
                surface (or the upper bounding surface), if they meet on the i-th dimension and intersect on all
                other dimensions. That is,
                1. max(Y1, i) = min(Y2, i) (or max(Y2, i) = min(Y1, i)) [for all i in cols./attrs. of data points]
                2. And min(Y1, j) < max(Y2, j) and max(Y1, j) > min(Y2, j) [for all j ≠ i]

                **Algorithm**:
                    1. Get the "preordered" list of leaves. [We are only concern about the leaves because at the end
                                                          these are the only set of nodes which contains all the
                                                          data points]
                    2. Initialize "mergeList", a blank list which will hold the pair of leaves which are touching
                                                            each-other by the above mentioned criterion.
                    3. for node_preOrderedPos in range(len(PreOrderedList_of_leaves)):
                        i. node = PreOrderedList_of_leaves[node_preOrderedPos]
                        ii. get the min and max of all the attribute values of the node.
                        iii. for all nxt_node in PreOrderedList_of_leaves[node_preOrderedPos:]:
                            a. for each attr_i in attributes:
                                **check for criteria 1. of the definition, if satisfies proceed to the step "a)"**
                                    a) for each attr_j in attributes except attr_i:
                                        **check for criteria 2. of the definition**
                            b. if both criteria 1 and 2 are satisfied a pair of leaf node:
                                a) Add the pair (node, nxt_node) to "mergeList".

                            end for

                        end for

                    4. Declare an empty dictionary named "touchingNodes", which will contain standardized list of
                            touching nodes for each node if a node is touching atleast one other node.
                    5. if "mergeList" is not empty:
                        a. invoke "normalizingMergeList()" with "mergeList",
                                            it will return the content of "touchingNodes"
                    6. Return "touchingNodes" to the calling function.



    normalizingMergeList:
        Parameters:
            mergeList: list of pair of nodes which are touching each-other by definition of "touching nodes".

        Operational Details:
            Create and return standardize list of touched nodes by each nodes respectively. This
            function takes care of "Commutative" and "Transitive" property to create standardize list of
            "touching nodes" for each node.
            Ex.: mergeList: [(A,B), (C,D)] ==> touchingNodes: {A: [B], B: [A], C: [D], D:[C]}
                 mergeList: [(A,B), (B,C)] ==> touchingNodes: {A: [B, C], B: [A, C], C: [A, B]}

            returns touchingNodes to the calling function.



    calcModifiedDataLengthForEachnode:
        Parameters:
            node: An instance of CLNode (which is part of a CLTree instance).

        Operational Details:
            This function initializes the "modifiedLength" property/attribute of CLNode.
            After initializing the "touching nodes" for each node, the idea is the node itself and all the
            touching node will **virtually** be considered a single node even though they are physically apart.
            So if we want to know how many data points are there in any node at any given point of time, we
            should include the length of data of "touching nodes" as well as virtually they are part of the
            same node.

            This is a recursive function and operation on the CLTree is "post-order"[processing the children
            first, then the node itself at any node] in nature.

            Algorithm:
                1. if the node is Leaf:
                    a. modfiedDataLength = own data length + ∑(data length of all "touching nodes")
                    b. node.modifiedDataLength = modifiedDataLength
                    c. return to the calling function
                2. else:
                    a. calculate the modified data length of the left child.
                    b. calculate the modified data length of the right child.
                    c. modifiedDataLength = left child modified data length + right child modified data length
                    d. node.modifiedDataLength = modifiedDataLength
                    e. return to the calling function

            This function set the values at the node level and doesn't return any value to the calling function.

    initializeTouchedNodes:
        Parameters:
            root: An instance of CLNode which is root of a CLTree instance.

        Operational Details:
            This function is driver for getting a standardized list of "touching nodes" by each leaf nodes and set
            the values for touchedNodes, modifiedLength at the CLNode instance level.
            [Refer to "identifyingTouchedRegions" to get a clear idea about "touching nodes"]
            Algorithm:
                1. get the standarized list of "touching nodes" by each leaf nodes by invoking
                                                                            "identifyingTouchedRegions".
                2. Assign the list of "touching nodes" by each leaf nodes at CLNode instance level one by one.
                3. calculate and initialize "modifiedLength" property/attribute of each CLNode by
                                                             invoking "calcModifiedDataLengthForEachnode".

            This function set the values at the CLNode instance level and
                                                doesn't return any value to the calling function.


    _accountForTouchedNodesForCentroids:
        Parameters:
            preOrderedListofLeaves: A list of leaves which was generated by traversing the initial CLTree in preorder.
            hyperRectangleCentroids: A python dictionary containing the initial centroid vector for each hyperrectangle
                                     created by each leaf.
            attributes: List of all the attributes in a constant order.

        Operational Details: This is an internal helper function which is invoked from _calcInitialCentroids
                             to calculate the resultant hyper-rectangle centroids by combining the
                             hyper-rectangle centroids of the "touching nodes" of each leaf and return the modified
                             collection of hyper-rectangle centroids to the calling function.

            Algorithm:
                1. Declare an empty python dictionary to store and return the new set of hyper-rectangle centroids.
                2. for each leaf in preOrderedListofLeaves:
                    i. get the "touching nodes" of the current leaf.
                    ii. if the current leaf node has any "touching nodes":
                        a. add the current leaf node at the beginning of the list of "touching nodes".
                        b. sort the current list of "touching nodes".
                        c. if we are seeing the sorted current list of "touching nodes" first time:
                            a. calculate the weighted average of all "hyperRectangleCentroids" of all the nodes
                                                            in current list of "touching nodes".
                            b. store the sorted current list of "touching nodes" as key and the result from previous
                                        step as the value for future use to the python dictionary declared at step 1.
                        d. if we have already seen the combination before, that mean this group is already accounted
                                        for and we can now move to the next leaf in the preOrderedListofLeaves.
                    iii. if the current leaf node doesn't have any "touching nodes", copy the current leaf node and
                                        its associated centroid to the python dictionary declared at step 1.
                3. end for
                4. Return the new set of hyper-rectangle centroids to the calling function.

    _calcInitialCentroids:
        Parameters:
            preOrderedListofLeaves: A list of leaves which was generated by traversing the initial CLTree in preorder.
            attributes: List of all the attributes in a constant order.

        Operational Details: This is an internal function with restricted access which is responsible for creating
                             initial centroids of all the hyper-rectangle created by leaves of original CLTree.
                             This function performs its operation by visiting each leaf node one by one from the
                             "preOrderedListofLeaves" list and it first calculate the difference between max and
                             min for each attribute one by one for each leaf, then divide all the amounts by 2 to
                             get the mid point and add those to the respective min values to get the centroid of
                             the hyper-rectangle created by the current leaf. After calculating the centroid of each
                             leaf separately we need to account for "touching nodes", we do that by invoking another
                             internal function "_accountForTouchedNodesForCentroids" with the initial list of
                             centroids and get back a modified list of centroids which is returned to the calling
                             function.

                            Algorithm:
                                1. Declare an empty python dictionary named "hyperRectangleCentroids".
                                2. for each leaf in "preOrderedListofLeaves":
                                    i. for each attribute in attributes:
                                        a. Get the min value of the current attribute of the current leaf and store it.
                                        b. Normalize the min by the "root" min and max of the current attribute.
                                        c. Get the max value of the current attribute of the current leaf and store it.
                                        d. Normalize the max by the "root" min and max of the current attribute.
                                    ii. Get the vector difference between max-vector and min-vector.
                                    iii. Calculate the centroid by adding the difference vector to the min vector.
                                    iv. Store the calculated centroid against the leaf for further use and move-on.
                                3. Invoke "_accountForTouchedNodesForCentroids" to account for "touching nodes" with
                                   the initial list of centroids along with "preOrderedListofLeaves" and "attributes"
                                   and get back a modified list of centroids which is returned to the calling function.

    initializeCentroids:
        Parameters:
            root: Instance of CLNode which is the root of underlying CLTree.

        Operational Details: This function is responsible to calculate the initial and original list of centroids and
                             return the result to the calling function.
                             Algorithm:
                                1. Validate the "root".
                                2. Invoke "preOrderTraversal" with the root and empty list to get the list of leaves
                                   which was generated by traversing the initial CLTree in preorder.
                                3. Get the list of attributes from the root.
                                4. Invoke "_calcInitialCentroids" with the pre-ordered leaves and the attributes we got
                                 at step 2 and 3 respectively and get back a python dictionary which contain the initial
                                   hyper-rectangle centroids before the pruning process begins and return it to the
                                   calling function.


    **Defination**: A unit vector is a vector of length 1, sometimes also called a direction vector.To find a
                     unit vector with the same direction as a given vector, we divide by the magnitude of the vector.

    _getUnitVectorOfRepresentationVectors:
        Parameters:
            representationVectors: A python dictionary, values of which are vectors.

        Operational Details: This is an internal utility function with restricted access. Given a collection of vectors
                             this function converts all the underlying vectors into "unit vector" individually and
                             return a dictionary with the same set of keys and respective unit-vectors of previously
                             provided vectors.

    _recalculateRepresentationVectors:
        Parameters:
            key1: tuple of node id(s)
            representationVector1: Current representation vector of the collection of nodes provided as key1.
            key2: tuple of node id(s)
            representationVector2: Current representation vector of the collection of nodes provided as key2.

        Operational Details: This is an internal utility function with restricted access. Given 2 collection of
                             node ids and their respective representation vectors this function calculates weighted
                             average of 2 representation vectors based on number of actual data instances each
                             collection encompasses and create a combined key from "key1" and "key2" and return the
                             newly calculated weighted average of two initial vectors as the associated value of the
                             new key to the calling function.


    _calcDataInstancesbyEachGroup:
        Parameters:
            listOfNodeIds: A list of tuples of node ids.

        Operational Details: This is an internal utility function with restricted access. Given a list of collection
                             of leaf node ids, this function calculates and returns how many actual data instances
                             are there by each collection of node ids.


    pruningTreeByMergingRepresentationVectors:
        Parameters:
            originalRepresentationVectors: A python dictionary containing the initial representation vectors for
                                           each initial group of leaf nodes [post accounted for "touching nodes"].
                                           Here representation vectors are the centroids of the hyper rectangle created
                                           by the leaf nodes[or weighted average of centroids of "touching nodes"].

            min_y: The minimum number of members required by each group of leaf nodes to be considered as an
                individual cluster.

        Operational Details:
            Base Algorithm:
            Agglomerative Hierarchical Clustering: It is a type of clustering algorithm which starts by treating
                each object as a singleton cluster. Next, pairs of clusters are successively merged until all
                clusters have been merged into one big cluster containing all objects. The result is a
                tree-based representation of the objects, named dendrogram.

            Quality Metric:
            1 of 3 different types Quality Metrics has been used in this implementation of the algorithm depending on
            the respective values of the instance level flag "useSilhouette" and "balancedPrune".

            Purity or Inv-Purity: intra-clusters-distance / inter-clusters-distance
            intra-cluster-distance and inter-cluster-distance are two very common quality metric is used to
            measure the quality of clusters or performance of the clustering algorithm. We always want to
            minimize intra-cluster-distance and maximize inter-clusters-distance for a given set of clusters.
            Here we have combined 2 term together and as we always want to minimize the numerator and maximize
            the denominator, our goal would be to minimize the value of our “Quality Metric”.

            varPurity/varInv-Purity: Standard deviation of cluster members of clusters ^ Purity or Inv-Purity
            As we have already established that why we want to minimize "Purity or Inv-Purity". At the "balancedPrune"
            mode we have an additional constraint which is variance/standard-deviation of number data points among
            clusters. We want to decrease "Purity or Inv-Purity" and "variance or standard-deviation",. By combining
            these two, we have created a new metric which gives emphasis on both metric at the same time. At the
            begining of an experiment when do not know anything or not have enough confidence it is recommended
            to do balanced clustering.

            silhouette: Silhouette refers to a method of interpretation and validation of consistency within clusters of
            data. The technique provides a succinct graphical representation of how well each object is classified.
            The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to
            other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the
            object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a
            high value, then the clustering configuration is appropriate. If many points have a low or negative value,
            then the clustering configuration may have too many or too few clusters.

            Defense/Protection against outliers:
            In clustering there is a common occurrence of a problem that there may be few data points scattered across
            the hyper-plane which are not close to any well-formed groups. Now the question is how to handle it?
            Here in our clustering scheme, we have an instance level parameter “min_y” which indirectly controls
            the number of clusters by defining how many minimum data points we would need to identify a group as a
            well-formed cluster. We don’t want to violate the integrity of well formed clusters to include some
            outliers here and there, so one of the better way of handling this kind of scenario is form a separate
            group of outliers by the name of “DEFAULT”, in this way we are protecting integrity of the well-formed
            clusters and also creating mutually exclusive and exhaustive clusters.
            We initiate the above-mentioned mechanism of “Protection against outliers” when the number of points
            which are not part of a well-formed group is less than the instance level parameter “min_y”.
            This way we are eliminating any possibility that they might form a well-formed cluster by merging
            one to another.


            This is the heart of the “PruneTreeByMergingCentroids”. At this point, our entire data space is
            divided into small groups, we have their representation vectors and another parameter, “min_y”.
            We will use the representation vectors as the source data to a hierarchical agglomerative clustering
            to group together small groups to form bigger groups. We will keep merging the small groups to form
            bigger groups until either all the remaining groups have “min_y” data points or total data points
            left to be part of bigger groups is less than “min_y”. In case of the later, we temporarily bind all
            the loose groups/data points which are not part of any qualified group and form a group.
            Then we calculate the statistics of the current settings and store it for future reference.
            Then we continue with the hierarchical agglomerative clustering until there is only 2 groups
            remaining but from now on after each iteration, we take a snapshot of the settings and the statistics
            for future reference.
            At the end we take all the scenarios, calculate the quality metric and use gradient tolerance to
            find out the best setting and return the best way to cluster to the calling function.

            Algorithm:
                1. Make a "deep copy" of initial representation vectors.
                2. calculate number of data points at each small groups and make a "deep copy" of the result.
                3. Get all the representation vectors in a way so that it can be directly used my the
                   "scipy.cluster.hierarchy" for hierarchical clustering and also, create reference of the
                    data points by using keys from the copy of the "originalRepresentationVectors" created at step 1.
                    So that later when we will revisit the hierarchical cluster tree from the bottom up manner
                    we can reference the original CLTree nodes from there.
                4. Min-Max normalization of the data which will be used for hierarchical clustering to remove any
                   potential bias.
                5. Hierarchical clustering on the normalized data.
                6. declare an empty dictionary to store all the intermediate results as we are about to revisit the
                   hierarchical cluster tree from bottom-up manner.
                7. While there are more than 2 groups:
                    i.   get the 2 vertices got combined.
                    ii.  get the actual node ids(or tuple of node ids) from the reference has been created at step 3.
                    iii. Create a new new combination from the 2 tuple we got back at previous step. It will be used
                         at id in to identify the combination of nodes.
                    iv.  create a new reference with the newly created id and delete previous references.
                    v.   Check if all the groups at this moment has data points more or equal to "min_y".
                         a. if so, take a snapshot of the current groups and move on to next iteration.
                    vi.  Otherwise, get 2 lists of groups which have atleast "min_y" data-points and groups which do not
                         respectively.
                         a. Check if total members of the groups which do not have atleast "min_y" data-points is
                            less than "min_y".
                            i> if so, combine all the non-qualified groups together to form a temporary default group.
                            ii> take a snapshot of the current groups and move on to next iteration.
                        b. Otherwise move-on to the next iteration.
                        c. end if
                    viii. end if
                8.  end while
                9.  Validation for no cluster found, it can happen due to bad value in parameters.
                10. Now it is the time to revisit all the snapshots we have taken during our visit to dendogram
                    to figure out the best possible setting by the chosen quality metric. 1 of 3 different types Quality
                    Metrics has been used in this implementation of the algorithm depending on the respective values of
                    the instance level flag "useSilhouette" and "balancedPrune".
                    i. If the instance level flag "useSilhouette" is True then the quality metric is "silhouette". In
                       this case our goal is to maximize the quality metric and maintain the setting which has the
                       highest quality metric indicator.
                    ii. If the instance level flag "balancedPrune" is True then the quality metric is "varPuriy". In
                        this case our goal is to minimize the quality metric and maintain the setting which has the
                        the lowest quality metric indicator.
                    iii. Otherwise, the quality metric is "purity". In this case our goal is to minimize the quality
                         metric and maintain the setting which has the lowest quality metric indicator.

                11. Once we have finalized a setting, we assign an unique cluster id to each group in the
                    finalized settings and also as a part of this process all the leaf nodes get assigned
                    to a cluster id based on which group they belong to.
                12. Return the best setting, final representation vectors and a map from node ids to cluster ids to the
                    calling function.


    _assigingClusterToInternalNodes:
        Parameters:
            scenario: Finalized cluster settings in python dictionary format. Where the keys are tuples made-up with
                     CLTree node ids.

        Operational Details:
            After finalizing the clusters and the leaf nodes each one encompasses, it is time to assign an unique
            cluster id to each cluster. In addition to this, the purpose of this function is to set the cluster
            flag of each leaf node one by one and assign the cluster-id to the node attribute based on which cluster
            the underlying leaf node is part of. At the same time create a reference from node ids to assigned
            cluster ids and return the same to the calling function.

    **Defination**: In computer science, tree traversal (also known as tree search) is a form of graph traversal
                    and refers to the process of visiting (checking and/or updating) each node in a tree data
                    structure, exactly once. Such traversals are classified by the order in which the nodes are
                    visited.
                A Generic Pre-Order Traversal Algorithm:
                    1. Check if the current node is empty or null.
                    2. Display/Store the data part of the root (or current node).
                    3. Traverse the left subtree by recursively calling the pre-order function.
                    4. Traverse the right subtree by recursively calling the pre-order function.


    preOrderTraversal:
        Parameters:
            node: Instance of CLNode class.
            preOrderedList: Current list of visited leaf nodes of underlying CLTree.

        Operational Details: This is an internal utility recursive function which returns the pre-order list of
                             visited **leaf nodes**. Unlike generic "Pre-Order Traversal Algorithm" this version
                             doesn't return list of all the nodes. It visits all the nodes but returns only the leaf
                             nodes in the order of traversal.

                             Algorithm:
                                    1. If the current node is a leaf node add the current node to the current list
                                         of visited leaf nodes and return the current list.
                                    2. If the current node is not a leaf node, get its children.
                                         i. Invoke another copy of "preOrderTraversal" with the left child and current
                                            list of visited leaf nodes and get back the modified current list of
                                            visited leaf nodes.
                                        ii. Invoke another copy of "preOrderTraversal" with the right child and current
                                            list of visited leaf nodes and get back the modified current list of
                                            visited leaf nodes.
                                        iii. Return the modified current list of visited leaf nodes to the
                                              calling function.


    preOrderTraversalofClusters:
        Parameters:
            node: Instance of CLNode class.
            preOrderedList: Current list of visited leaf nodes of underlying CLTree which are part of some cluster.

        Operational Details: This is an internal utility recursive function which returns the pre-order list of
                            visited **leaf nodes which are part of some cluster**. Unlike generic
                            "Pre-Order Traversal Algorithm" this version doesn't return list of all the nodes.
                            It visits all the nodes but returns only the leaf which are part of some cluster
                            nodes in the order of traversal.

                            Algorithm:
                                      1. If the current node is part of some cluster add the current node to the
                                         current list of visited "includedInCluster" nodes and return the current list.
                                      2. If the current node is not "includedInCluster", then get its children.
                                         i. If it is a leaf node, just return the unmodified current list of visited
                                            leaf nodes which are part of some cluster.
                                        ii. Invoke another copy of "preOrderTraversalofClusters" with the left child
                                             and current list of visited "includedInCluster" nodes and get back the
                                             modified current list of visited "includedInCluster" nodes.
                                        iii. Invoke another copy of "preOrderTraversal" with the right child and
                                              current list of visited "includedInCluster" nodes and get back the
                                              modified current list of visited "includedInCluster" nodes.
                                        iv. Return the modified current list of visited "includedInCluster" nodes to
                                             the calling function.

    leafPruning:
        Parameters:
            node: Instance of CLNode class.

        Operational Details: This is an internal utility recursive function which operate in a bottom-up fashion.
                             At each level/node this function checks if 2 children of a parent are part of the same
                             cluster, if so we can safely assign the cluster at the parent level and by doing so,
                             we are optimizing the search space by reducing it by one level at a time.

    _whichChild:
        Parameters:
            node: Instance of CLNode class.

        Operational Details: This is an internal utility function with restricted access, which, given a node,
                             return a flag to indicate if the current node is left or right child of its parent.


    resetAllPruneNodes:
        Parameters:
            node: Instance of CLNode class.

        Operational Details: This is an utility recursive function which resets all the changes has been made
                             at node level of the underlying CLTree as part of the current pruning process and revert
                             back to its original structure.This function is called when doing grid search to find
                             optimal values for the set of hyperparameters.


    _calcClusterMemebersVariance:
        Parameters:
            totalDataInstancesbyEachGroup: A python dictionary which contains data instances encompassed by each group
                                           of leaf nodes.

        Operational Details: This is an internal utility function with restricted access which is called from
                             "clusterStatistics". This function returns the mean of the number of data points
                             accross the clusters and the variance among the number of data points accross the clusters

    clusterStatistics:
        Parameters:
            originalRepresentationVectors: Python dictionary consist of original representation vectors. Keys are the
                                           original leaf nodes after "touching nodes" are accounted for and values
                                           are respective representation vectors before
                                           "pruningTreeByMergingRepresentationVectors" was invoked.
            finalRepresentationVectors: Python dictionary consist of final representation vectors. Keys are the final
                                        groups of leaf nodes in tuple format which together forms invidual clusters
                                        and values are respective representation vectors after
                                        "pruningTreeByMergingRepresentationVectors" operation is completed.
            finalRepresentationVectorsToCluster: Python dictionary consist of mapping from collection of leaf nodes
                                                 to assigned respective cluster ids.
            totalDataInstancesbyEachGroup: Python dictionary consist of mapping from group of leaf nodes and total
                                           number of data instances by each group.


        Operational Details:
            This function is responsible to calculate and return the key statistics including the quality metric of
            choice which will be used to asses the quality of clustering. In this particular implementation the
            potential quality metrics are, purity or inv-purity, Formula:intraClusterDistance/interClusterDistance
            and varPurity or var-invPurity,
            Formula: Standard-deviation among the number of data points at each cluster ^ purity or inv-purity and
            silhouette indicator, Formula: mean of silhouette constants of all the clusters in a setting respectively.
            Apart from the quality metric by itself, depending on the value of the instance level flag "useSilhouette"
            this function also returns the "intra-cluster-distance", "inter-cluster-distance", "mean-members" across the
            clusters  and "data-points" at each cluster or calculated Silhouette Constants and "data-points" at each
            cluster under the provided setting.

            Algorithm:
                1. If the instance level variable "useSilhouette" is True then the quality metric is set to "silhouette"
                   otherwise if the instance level variable "balancedPrune" is True set the quality metric as
                   "varPurity" else set the quality metric as "purity".
                2. Get the unit vector of each original and final representation vectors.
                3. Create a map/dictionary for each final individual group of leaf nodes which was created
                   by merging a subset of original set of leaf nodes. It would be useful at the time of calculating
                   "intra-cluster-distance" or to determine a(i) in case of "Silhouette" constant calculation for
                   cluster [ref:  https://en.wikipedia.org/wiki/Silhouette_(clustering)].
                4. If the value of instance level variable "useSilhouette" is NOT True then,
                    i.    Create all possible pair of clusters. It would be useful at the time of calculating
                          "inter-cluster-distance".
                    ii.   Invoke an internal function named, "_calcClusterStatistics" to get the
                          "inter-cluster-distance" and "intra-cluster-distance".
                    iii.  calculate the purity or inv-purity by "intra-cluster-distance"/"inter-cluster-distance".
                    iv.   Invoke "_calcClusterMemebersVariance" to get the mean of the number of data points across the
                          clusters and the variance among the number of data points accross the clusters.
                    v.    Calculate "varPurity" using the information at step 7 and 6.
                    vi.   Put all the statistics together in "result".
                    vii.  Add "argminMetric" in the "result", according to the value of quality metric decided at step 1
                    viii. Return the result to the calling function.
                5. If the value of instance level variable "useSilhouette" is True then,
                    i.   For each cluster create a list of all possible combinations by including one of remaining
                         clusters at a time and move on to the next cluster and continue to do the same thing until all
                         the clusters are covered. These combinations would be used in the calculation of the pair wise
                         distance among all clusters to determine b(i)
                         [ref:  https://en.wikipedia.org/wiki/Silhouette_(clustering)] of the "Silhouette" for each
                         cluster.
                    ii.  Invoke an internal function named "_calcSilhouetteConstants" to get back the individual
                         Silhouette constants for all the clusters in the current setting.
                    iii. To calculate the quality metric of the current setting, take the mean of all individual
                         Silhouette constants.
                    iv.  Put all the statistics together in "result".
                    v.   Return the result to the calling function.

    _calcSilhouetteConstants:
        Parameters:
            individualPairWiseFinalRepresentationVectors: A python dictionary containing list of all 2 members
                                                          combinations of clusters for each cluster. Ex.
                                                          Suppose you have 3 clusters, A, B and C then,
                                                          {'A':['A', 'B'], ['A', 'C'],
                                                           'B':['B', 'A'], ['B', 'C'],
                                                           'C':['C', 'A'], ['C', 'B']}

            finalRepresentationVectors: Python dictionary consist of final representation vectors. Keys are the final
                                        groups of leaf nodes in tuple format which together forms individual clusters
                                        and values are respective representation vectors after
                                        "pruningTreeByMergingRepresentationVectors" operation is completed.

            originalRepresentationVectors: Python dictionary consist of original representation vectors. Keys are the
                                           original leaf nodes after "touching nodes" are accounted for and values
                                           are respective representation vectors before
                                           "pruningTreeByMergingRepresentationVectors" was invoked.

            finalToOriginalRepresentationVectorsMap: A python dictionary for each final individual group of leaf nodes
                                                     which was created by merging a subset of original set of
                                                     leaf nodes. These combinations would be used in the calculations of
                                                     the a(i)/pair-wise distances from one point within a custer to all
                                                     all the other points of the same cluster for each cluster.

        Operational Details:
            This is an internal function which is called from "clusterStatistics", is responsible to
            do the actual calculations required to calculate the Silhouette constants for each clusters. This function
            is an implementation of the content @ https://en.wikipedia.org/wiki/Silhouette_(clustering).
            1. For the calculation of a(i) of the Silhouette constant for each cluster the reference point is the
               centroid of each final clusters and other points are the representation vectors of all the leaves
               encompassed by the underlying cluster.
            2. For the calculation of b(i) of the Silhouette constant for each cluster the reference point is the
               centroid of respective final clusters and other points are the centroids of all the other final clusters.
            3. Silhouette constant is calculated by the formula: {b(i) - a(i)} / {max(b(i), a(i))} for each cluster.
            4. The values calculated at step 3 is returned to the function "clusterStatistics".

    _calcClusterStatistics:
        Parameters:
            allCombosOfFinalRepresentationVectors: All possible pair of clusters' keys. It would be useful at the time
                                                   of calculating "inter-cluster-distance"

            finalRepresentationVectors: Python dictionary consist of final representation vectors. Keys are the final
                                        groups of leaf nodes in tuple format which together forms invidual clusters
                                        and values are respective representation vectors after
                                        "pruningTreeByMergingRepresentationVectors" operation is completed.

            originalRepresentationVectors: Python dictionary consist of original representation vectors. Keys are the
                                            original leaf nodes after "touching nodes" are accounted for and values
                                            are respective representation vectors before
                                            "pruningTreeByMergingRepresentationVectors" was invoked.

            finalToOriginalRepresentationVectorsMap: A python dictionary for each final individual group of leaf nodes
                                                     which was created by merging a subset of original set of
                                                     leaf nodes. It would be useful at the time of calulating
                                                     "intra-cluster-distance".

        Operational Details:
            This is an internal function which is called from "clusterStatistics", is responsible to
             do the actual calculations required to calculate various cluster statistics including
             "intra-cluster-distance" and "inter-cluster-distance" and return those to the calling function.



    _calcEuclidianDistance:
        Parameters:
            a: vector 1.
            b: vector 2.

        Operational Details:
            This is an utility function which calculates and returns the euclidean distance between 2 vectors.

    getClusterID:
        Operational Details:
            This function increment and get the current value of an instance level variable and then concatenate
            the value with some predefined string to create and the return an unique id for a cluster.

    pruningRedundantNodes:
        Parameters:
            root: The root of the instance level CLTree.

        Operational Details:
            This is an utility function. Purpose of this function is to make the search space of CLTree more efficient
            after "leafPruning" has happened. At core CLTree divides the data-set at any level like a binary search tree
            Suppose 2 leaf nodes are 1 generation apart, both are the same type of child of their respective parent and
            at both respective parent level the cut has happened on the same attribute, then we can safely remove
            one level from the existing CLTree.
            **Note**: This method changes the structure of the core CLTree, so after this method is invoked the
                      underlying CLTree can't be reused in grid-search or restored back to its original form.
                      This method/function therefore should be only called during the final and permanent pruning
                      process.
            Each leaf node in CLTree holds an unique set of data, as we are removing one level, we would need to merge
            the data of respective leaf node to the data set of the other leaf node which will represent the both leaves
            together.
            This is a bottom-up iterative procedure.

    _recalculateDepthOfEachNodes:
        Parameters:
            node: CLNode instance. Started with CLTree root.
            depth: Current Depth of the node. Started with 0.

        Operational Details:
            This is a recursive procedure, the purpose of the procedure is recalculate the depth of all the nodes
            of the CLTree after it has gone through some structural changes. It updates the depth at the node level.

    _merging2Datasets:
        Parameters:
            dataset1: Instance of CLTreeModules.Data class associated with one CLNode instance.
            dataset2: Instance of CLTreeModules.Data class associated with another CLNode instance.

        Operational Details:
            This is an internal function called from "pruningRedundantNodes" to merge dataset of 2 nodes together when
            reducing a level. Some basic validation is required before the merge. Ex.: Order of the columns
            name of all the columns and data-types of all the columns need to be same among 2 participating dataset.
            If all the validations are fruitful then we create a separate instance of CLTreeModules.Data class
            by using the combined data-set of data-set1 and data-set2 and return the newly created instance back to
            calling function.

pruneByGridSearch_Centroid:
    Parameters:
        cltree: The original CLTree instance on which the whole puning mechanism will be performed.
        min_y: The "min_y" dictates the minimum number of data points required for a group to be considered as a cluster
        data_length: Total number of the data points on which the clustering is happening.
        prefixString: predefined string for cluster id.
        balancedPrune: It is a boolean flag, value of which indicates if we want to create a "balanced" clusters
                       in terms of number of data points each cluster get.
        useSilhouette: Is a Binary flag variable, default of it is False. If the value of this flag is True then the
                       "Silhouette" constant indicator will be calculated to determine the optimal number of clusters
                        from a dataset otherwise [avg. intra-cluster distance/avg. inter-cluster distance] will be used
                        to determine the optimal number of clusters from a data-set.

    Operational Details: This is method to grid search on various parameter values required for creating cluster
                         by pruning the initial CLTree or merging the leaf nodes to form clusters.
                         In this implementation, "min_y" is the only manually provided value and also a
                         hyper parameter. To get the most optimal value of "min_y" which will provide the
                         best value for the cost function, we do a grid search on 10 equally spaced values of
                         "min_y" starting from "min_y" - 10% of "min_y" to "min_y". At each value we create a
                         instance of "PruneTreeByMergingCentroids" with the original "cltree" and current value of
                         "min_y". At the end of each pruning procedure, we store the current
                         qulaity metric with the current setting for future reference. (Here in this implementation
                         the potential quality metrics are "purity"[actually inverse purity.] or
                         "varPurity" [actually std. ^ inverse purity.], which are we are trying to minimize or
                         "silhouette" indicator which we are trying to maximize.
                         Note: An single call to the function focuses on a single quality metric depending on the values
                         of parameters, "useSilhouette" and/or "balancedPrune".)
                         Then move to the new value of "min_y". At the end when we are done with testing all the
                         potential values of "min_y". We take the value of "min_y" where the quality metric is best
                         to the final pruning with "searchMode" set to False. We set the base version of the pruning
                         algorithm and return the base version and the result to the calling function. After the final
                         pruning all the required changes has happened to the CLTree and CLNode level.
"""

import copy
import logging
import scipy.cluster.hierarchy as shc
import numpy as np
from itertools import product, combinations
from CLTreeModules import *
from ClusterTree_Utility import Queue

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PruneTreeByMergingCentroids(object):
    def __init__(self, prefixString, cltree, balancedPrune=False, useSilhouette=False):
        self.cltree = cltree
        self.root = self.cltree.getRoot()
        self.version = None
        self.balancedPrune = balancedPrune
        if isinstance(prefixString, str):
            self.prefixString = prefixString
        else:
            self.prefixString = str(prefixString)

        self.useSilhouette = useSilhouette

    def prune(self, min_y, searchMode=True):
        self.min_y = max(1, min_y)
        self.maxSoFar = -1
        self.version = 2.0
        self.totalDataInstancesbyEachGroup = None
        self.finalRepresentationVectors = None
        self.finalRepresentationVectorsToCluster = None
        self.originalRepresentationVectors = None
        self.initializeTouchedNodes(self.root)
        self.originalRepresentationVectors = self.initializeCentroids(self.root)


        totalDataInstancesbyEachGroup, finalRepresentationVectors, finalRepresentationVectorsToCluster = \
                        self.pruningTreeByMergingRepresentationVectors(self.originalRepresentationVectors, self.min_y)

        if totalDataInstancesbyEachGroup is None:
            return None

        self.totalDataInstancesbyEachGroup = totalDataInstancesbyEachGroup
        self.finalRepresentationVectors = finalRepresentationVectors
        self.finalRepresentationVectorsToCluster = finalRepresentationVectorsToCluster
        self.leafPruning(self.root)

        result = self.clusterStatistics(copy.deepcopy(self.originalRepresentationVectors),
                                        copy.deepcopy(self.finalRepresentationVectors),
                                        copy.deepcopy(self.finalRepresentationVectorsToCluster),
                                        copy.deepcopy(self.totalDataInstancesbyEachGroup))


        if not searchMode:
            self.pruningRedundantNodes(self.cltree.root)
            result2 = self.clusterStatistics(self.originalRepresentationVectors,
                                             self.finalRepresentationVectors,
                                             self.finalRepresentationVectorsToCluster,
                                             self.totalDataInstancesbyEachGroup)

            logger.info("Search Mode: False.\
            Note: The underlying Tree structure got modified in the process of pruning to make the search space optimized!\
            'min_y' = {}, 'result' = \n{}".format(min_y, result2))
            return result2

        else:
            logger.info("Searching for 'min_y'. Current 'min_y' = {}, 'result' = \n{}".format(min_y, result))
            return result

    def identifyingTouchedRegions(self, root):
        # Get back a preorder traversal list of all leaf nodes.
        prunePreOrderedList = self.preOrderTraversal(root, list())
        mergeList = []
        for i in range(len(prunePreOrderedList)):
            node_i = self.cltree.nodes[prunePreOrderedList[i]]
            data_i = node_i.dataset
            attributes_i = data_i.attr_names
            attributes_idx_i = data_i.attr_idx
            maxvalues_i = data_i.max_values
            minvalues_i = data_i.min_values
            for j in range(i + 1, len(prunePreOrderedList)):
                node_j = self.cltree.nodes[prunePreOrderedList[j]]
                data_j = node_j.dataset
                attributes_j = data_j.attr_names
                attributes_idx_j = data_j.attr_idx
                maxvalues_j = data_j.max_values
                minvalues_j = data_j.min_values
                attr_ij = None
                flag_ij = False
                for attr in attributes_i:
                    ii = attributes_idx_i[attr]
                    min_i = minvalues_i[ii]
                    max_i = maxvalues_i[ii]
                    jj = attributes_idx_j[attr]
                    min_j = minvalues_j[jj]
                    max_j = maxvalues_j[jj]
                    if min_i == max_j or max_i == min_j:
                        flag_ij = True
                        attr_ij = attr
                        break

                if flag_ij:
                    otherAttrs_i = [attr for attr in attributes_i if attr != attr_ij]

                    for attr in otherAttrs_i:
                        ii = attributes_idx_i[attr]
                        min_i = minvalues_i[ii]
                        max_i = maxvalues_i[ii]
                        jj = attributes_idx_j[attr]
                        min_j = minvalues_j[jj]
                        max_j = maxvalues_j[jj]
                        if (min_i < max_j and max_i > min_j) or (min_j < max_i and max_j > min_i):
                            continue
                        else:
                            flag_ij = False
                            break
                    if flag_ij:
                        logger.info("Adding to merge list: {}".format((node_i.nodeId, node_j.nodeId)))
                        mergeList.append((prunePreOrderedList[i], prunePreOrderedList[j]))

        touchingNodes = dict()
        logger.info("Printing MergeList: {}".format(mergeList))
        if len(mergeList) > 0:
            touchingNodes = self.normalizingMergeList(mergeList)
        logger.info("Printing Touching Nodes: {}".format(touchingNodes))
        return touchingNodes

    def normalizingMergeList(self, mergeList):
        touchingNodes = dict()
        for nodeId_i, nodeId_j in mergeList:
            list_i = touchingNodes.get(nodeId_i, set())
            list_j = touchingNodes.get(nodeId_j, set())
            list_i.add(nodeId_j)
            list_j.add(nodeId_i)
            touchingNodes[nodeId_i] = list_i
            touchingNodes[nodeId_j] = list_j

        for nodeId in touchingNodes.keys():
            queue = Queue()
            visited = set(touchingNodes[nodeId])
            for i in visited:
                queue.enqueue(i)
            while not queue.isEmpty():
                n = queue.dequeue()
                for node in touchingNodes[n]:
                    if node not in visited:
                        queue.enqueue(node)
                        visited.add(node)

            visited = visited - {nodeId}
            touchingNodes[nodeId] = visited

        return touchingNodes

    def calcModifiedDataLengthForEachnode(self, node):
        if node.isLeaf():
            node.setPruneState(True)
            modifiedDataLength = node.dataset.length()
            if node.getTouchedNodes() is not None:
                touchedNodes = node.getTouchedNodes()
                for i in touchedNodes:
                    modifiedDataLength += self.cltree.nodes[i].dataset.length()
            node.setModifiedDataLength(modifiedDataLength)
            return
        else:
            node_L = node.getLeft()
            node_R = node.getRight()
            self.calcModifiedDataLengthForEachnode(node_L)
            self.calcModifiedDataLengthForEachnode(node_R)
            modifiedDataLength = node_L.getModifiedDataLength() + node_R.getModifiedDataLength()
            node.setModifiedDataLength(modifiedDataLength)
            return

    def initializeTouchedNodes(self, root):
        touchingNodes = self.identifyingTouchedRegions(root)
        for nodeId in touchingNodes.keys():
            node = self.cltree.nodes[nodeId]
            touchedNodes = touchingNodes[nodeId]
            for i in touchedNodes:
                node.addTouchedNodes(i)

        self.calcModifiedDataLengthForEachnode(self.root)
        return

    def _accountForTouchedNodesForCentroids(self, preOrderedListofLeaves, hyperRectangleCentroids, attributes):
        centroids = {}
        for i in preOrderedListofLeaves:
            node = self.cltree.nodes[i]
            if node.getTouchedNodes() is not None:
                touchedNodes = node.getTouchedNodes()
                touchedNodes.insert(0, i)
                touchedNodes = tuple(sorted(touchedNodes))
                if touchedNodes in centroids.keys():
                    continue
                else:
                    instances = np.zeros(len(touchedNodes))
                    c = np.zeros((len(touchedNodes), len(attributes)))
                    for ii in range (len(touchedNodes)):
                        c_ii = hyperRectangleCentroids[touchedNodes[ii]]
                        instances[ii] = self.cltree.nodes[ii].dataset.length()
                        c[ii,] = c_ii
                    totalInstances = np.sum(instances)
                    normalizedInstances = instances/totalInstances
                    normalizedInstances = normalizedInstances.reshape((len(touchedNodes),1))*\
                                            np.ones((len(touchedNodes),len(attributes)))
                    c = c/normalizedInstances
                    centroid = np.sum(c, axis=0)
                    centroids[touchedNodes] = centroid
            else:
                key = tuple([i])
                centroids[key] = np.array(hyperRectangleCentroids[i])

        return centroids

    def _calcInitialCentroids(self, preOrderedListofLeaves, attributes):
        eps = 0.0000001
        hyperRectangleCentroids = {}
        rootDataset = copy.deepcopy(self.cltree.getRoot().dataset)  # Make a copy of the original data set
        for i in preOrderedListofLeaves:
            node = self.cltree.nodes[i]
            minVector = np.zeros(len(attributes))
            maxVector = np.zeros(len(attributes))
            hyperRectangleDimensions = np.zeros(len(attributes))
            centroid = np.zeros(len(attributes))
            for ii in range(len(attributes)):
                attr = attributes[ii]
                rootMin = rootDataset.get_min(attr)  # Min-Max normalization
                rootMax = rootDataset.get_max(attr)  # Min-Max normalization
                minV = node.dataset.get_min(attr)
                minV = (minV - rootMin) / (rootMax - rootMin + eps)  # Min-Max normalization
                minVector[ii] = minV
                maxV = node.dataset.get_max(attr)
                maxV = (maxV - rootMin) / (rootMax - rootMin + eps)  # Min-Max normalization
                maxVector[ii] = maxV

            hyperRectangleDimensions = maxVector - minVector
            deltaDistanceForCentroid = hyperRectangleDimensions/2
            centroid = minVector + deltaDistanceForCentroid
            hyperRectangleCentroids[i] = centroid

        centroids = self._accountForTouchedNodesForCentroids(
                    preOrderedListofLeaves, hyperRectangleCentroids, attributes)

        return centroids

    def initializeCentroids(self, root):
        assert root.depth == 0, logger.error("Depth of the root of the CLTree should be zero(0)!,\
        either the the 'depth' of each node in CLTree is not assigned properly or the node passed is not root!")
        preOrderedListofLeaves = self.preOrderTraversal(root, [])
        attributes = root.dataset.attr_names
        centroids = self._calcInitialCentroids(preOrderedListofLeaves, attributes)
        return centroids

    def _getUnitVectorOfRepresentationVectors(self, representationVectors):
        eps = 0.0000001
        assert isinstance(representationVectors, dict), logger.error("This function is expecting a key-value pair!")
        for rv in representationVectors.keys():
            representationVector = representationVectors[rv]
            if not isinstance(representationVector, np.ndarray):
                representationVector = np.array(representationVector)
            magnitude = np.sqrt(np.sum(np.square(representationVector)))
            representationVector = representationVector / (magnitude+eps)
            representationVectors[rv] = representationVector
        return representationVectors

    def _recalculateRepresentationVectors(self, key1, representationVector1, key2, representationVector2):
        dataInstances_1 = 0
        dataInstances_2 = 0
        for i in key1:
            dataInstances_1 += self.cltree.nodes[i].dataset.length()
        for j in key2:
            dataInstances_2 += self.cltree.nodes[j].dataset.length()
        totalDataInstances = dataInstances_1 + dataInstances_2
        if not isinstance(representationVector1, np.ndarray) or not isinstance(representationVector2, np.ndarray):
            representationVector1 = np.array(representationVector1)
            representationVector2 = np.array(representationVector2)

        representationVector = representationVector1 * (dataInstances_1 / totalDataInstances) + \
                                            representationVector2 * (dataInstances_2 / totalDataInstances)
        if not isinstance(key1, tuple) or not isinstance(key2, tuple):
            key1 = tuple(key1)
            key2 = tuple(key2)

        key = key1 + key2
        key = tuple(sorted(key))
        assert len(key) == len(set(key)), "ERROR! Some node counted twice at the time of pruning!"
        keyRepresentationVector = {key: representationVector}

        return keyRepresentationVector, totalDataInstances

    def _calcDataInstancesbyEachGroup(self, listOfNodeIds):
        totalDataInstancesbyEachGroup = {}
        for i in listOfNodeIds:
            val = 0
            for ii in i:
                val += self.cltree.nodes[ii].dataset.length()

            totalDataInstancesbyEachGroup[i] = val

        return totalDataInstancesbyEachGroup

    def pruningTreeByMergingRepresentationVectors(self, originalRepresentationVectors, min_y):
        representationVectors = copy.deepcopy(originalRepresentationVectors)
        originalTotalDataInstancesbyEachGroup = self._calcDataInstancesbyEachGroup(list(representationVectors.keys()))
        totalDataInstancesbyEachGroup = copy.deepcopy(originalTotalDataInstancesbyEachGroup)

        rows = len(representationVectors)
        cols = list(representationVectors.values())[0].shape[0]
        data= np.zeros((rows, cols))
        row = 0
        mapping_keyToPos = {}
        mapping_posToKeys = {}
        vertices = {}
        for key in representationVectors.keys():
            val = representationVectors[key]
            data[row, :] = val
            mapping_keyToPos[key] = row
            mapping_posToKeys[row] = key
            vertices[row] = tuple([row])
            row +=1
        data = (data - data.min(0)) / (data.ptp(0)+0.00001) # Here, x.ptp(0) returns the "peak-to-peak"
        # (i.e. the range, max - min) along axis 0. This normalization also guarantees that the minimum value in
        # each column will be 0.

        # Agglomerative clustering on whole data.
        tree = shc.linkage(data, method='ward')
        # dend = shc.dendrogram(tree) # If you want to see the graphical representation, feel free to uncomment it.

        # Now we will visit the tree from bottom-up and retrieve information
        # Neeed to modify totalDataInstancesbyEachGroup and mapping_posToKeys as we clim-up the tree
        # Once all the groups reach the minimum number of datapoints required, take snapshot and continue further
        # till the point when only 2 groups left.
        # Later we will use this snap-shots to determine what is the ideal number of groups/clusters for the
        # underlying min_y.

        snapshot = {}
        i = 0
        while len(vertices)>2:
            m = max(vertices.keys())
            vx1 = tree[i, 0]
            vx2 = tree[i, 1]
            newVxKey = m + 1

            vx1Val = vertices.get(int(vx1))
            vx2Val = vertices.get(int(vx2))
            newVxVal = vx1Val + vx2Val

            node1 = mapping_posToKeys.get(int(vx1))
            node2 = mapping_posToKeys.get(int(vx2))
            newNode = tuple(sorted(node1 + node2))

            pop1 = totalDataInstancesbyEachGroup.get(node1)
            pop2 = totalDataInstancesbyEachGroup.get(node2)
            newPopulation = pop1+pop2


            del vertices[vx1]
            del vertices[vx2]
            vertices[newVxKey] = newVxVal

            del mapping_posToKeys[vx1]
            del mapping_posToKeys[vx2]
            mapping_posToKeys[newVxKey] = newNode

            del totalDataInstancesbyEachGroup[node1]
            del totalDataInstancesbyEachGroup[node2]
            totalDataInstancesbyEachGroup[newNode] = newPopulation

            if min(totalDataInstancesbyEachGroup.values())>=min_y:
                snapshot[len(vertices)] = copy.deepcopy(totalDataInstancesbyEachGroup)
            else:
                lengths = list(totalDataInstancesbyEachGroup.values())
                lengths = [l for l in lengths if l < min_y]
                if sum (lengths) < min_y:
                    notQualifiedGroups = {k: v for k,v in totalDataInstancesbyEachGroup.items() if v<min_y}
                    qualifiedGroups = {k: v for k,v in totalDataInstancesbyEachGroup.items() if v>=min_y}
                    
                    newNode = tuple()
                    length = 0 
                    for n in notQualifiedGroups.keys():
                        newNode = newNode + n
                        length = length + notQualifiedGroups[n]
                    newNode = tuple(sorted(newNode))
                    qualifiedGroups[newNode] = length
                    snapshot[len(qualifiedGroups) - 0.01] = copy.deepcopy(qualifiedGroups)
            i+=1


        # VALIDATION FOR NO CLUSTER FOUND
        if max(totalDataInstancesbyEachGroup.values()) < min_y:
            return None, None, None

        # Revisiting the snapshots taken while traversing.
        if not self.useSilhouette:
            bestResult = float('inf')
        else:
            bestResult = float('-inf')

        bestScenario = None
        bestFinalVectors = None

        for key in snapshot.keys():
            scenario = snapshot[key]
            scenario_FinalRepresentationVectors = {}
            tmp_scenario_finalRVToCluster = {}
            clusterNo = 0
            for representationVectorKey_F in scenario.keys():
                tmp_scenario_finalRVToCluster[representationVectorKey_F] = 'Cluster_'+str(clusterNo)
                clusterNo += 1

                l = []
                for representationVectorKey_O in representationVectors.keys():
                    if len(set(representationVectorKey_O) - set(representationVectorKey_F)) == 0:
                        l.append(representationVectorKey_O)
                if len(l) == 0:
                    raise ValueError("The Node keys used to construct the original tree mismatched\
                    from the keys used to group the leaf nodes!")

                elif len(l) == 1:
                    scenario_FinalRepresentationVectors[representationVectorKey_F] = representationVectors[l[0]]
                else:
                    origKey_1 = l[0]
                    origKey_2 = l[1]
                    origRV_1 = representationVectors[origKey_1]
                    origRV_2 = representationVectors[origKey_2]
                    key_RV, _ = self._recalculateRepresentationVectors(origKey_1, origRV_1, origKey_2, origRV_2)
                    for ii in range(2,len(l)):
                        origKey_ii = l[ii]
                        origRV_ii = representationVectors[origKey_ii]
                        key_RV, _ = self._recalculateRepresentationVectors(list(key_RV.keys())[0],
                                                                           list(key_RV.values())[0],
                                                                            origKey_ii, origRV_ii)
                    scenario_FinalRepresentationVectors[representationVectorKey_F] = list(key_RV.values())[0]

            result = self.clusterStatistics(originalRepresentationVectors = copy.deepcopy(representationVectors),
                                            finalRepresentationVectors = copy.deepcopy(scenario_FinalRepresentationVectors),
                                            finalRepresentationVectorsToCluster=copy.deepcopy(tmp_scenario_finalRVToCluster),
                                            totalDataInstancesbyEachGroup = copy.deepcopy(scenario))

            if not self.useSilhouette:
                logger.info("Min_Y: {}, Scenario: {},  #Clusters: {}, argminMetric: {}".format(min_y, scenario.values(),\
                                                                                             key, result['argminMetric']))
                # Going more to less number of clusters.
                # If we make 10000 clusters from a 10000 data points the value we are optimizing will be the lowest.
                # Our goal is to make sensible and manageable number of clusters where each cluster has atleast certain
                # amount of data points.

                if result['argminMetric'] <= bestResult:
                    bestResult = result['argminMetric']
                    bestScenario = scenario
                    bestFinalVectors = scenario_FinalRepresentationVectors
            else:
                logger.info("Min_Y: {}, Scenario: {},  #Clusters: {}, silhouette-indicator: {}".format(min_y,
                                                                        scenario.values(),key, result['silhouette']))
                if result['silhouette'] >= bestResult:
                    bestResult = result['silhouette']
                    bestScenario = scenario
                    bestFinalVectors = scenario_FinalRepresentationVectors

        finalRepresentationVectorsToCluster = self._assigingClusterToInternalNodes(bestScenario)

        return bestScenario, bestFinalVectors, finalRepresentationVectorsToCluster


    def _assigingClusterToInternalNodes(self, scenario):
        repVecKeyToCluster = {}
        for repVecKey in scenario.keys():
            if scenario[repVecKey] >= self.min_y:
                clusterID = self.getClusterID()
            else:
                clusterID = self.prefixString + '_' + "cluster_"+"DEFAULT"
            
            repVecKeyToCluster[repVecKey] = clusterID

            for nodeId in repVecKey:
                node = self.cltree.nodes[nodeId]
                node.includedInCluster = True
                node.setPruneState(True)
                node.clusterId = clusterID

        return repVecKeyToCluster

    def preOrderTraversal(self, node, preOrderedList):
        if node.isPrune():
            preOrderedList.append(node.getID())
            return preOrderedList

        elif node.isLeaf():
            if not node.isPrune():
                node.setPruneState(True)
            preOrderedList.append(node.getID())
            return preOrderedList

        else:
            children = node.getChildNodes()
            if len(children) == 1 and not node.isPrune():
                children[0].setPruneState(True)
                node.setPruneState(True)
                preOrderedList.append(node.getID())
                return preOrderedList

            else:
                preOrderedList = self.preOrderTraversal(children[0], preOrderedList)
                preOrderedList = self.preOrderTraversal(children[1], preOrderedList)

                return preOrderedList

    def preOrderTraversalofClusters(self, node, preOrderedListofClusters):
        if node.includedInCluster:
            preOrderedListofClusters.append((node.parent.attribute, node.clusterId, node))
            return preOrderedListofClusters
        else:
            children = node.getChildNodes()
            if len(children) == 0:
                return preOrderedListofClusters

            preOrderedListofClusters = self.preOrderTraversalofClusters(children[0], preOrderedListofClusters)
            if len(children) == 2:
                preOrderedListofClusters = self.preOrderTraversalofClusters(children[1], preOrderedListofClusters)
            return preOrderedListofClusters

    def leafPruning(self, node):
        if node.includedInCluster:
            return True
        else:
            children = node.getChildNodes()
            lc = self.leafPruning(children[0])
            rc = self.leafPruning(children[1])
            if lc and rc:
                lcId = children[0].clusterId
                rcId = children[1].clusterId
                if lcId == rcId:
                    node.includedInCluster = True
                    node.clusterId = rcId
                    node.setPruneState(True)
                    return True
                else:
                    return False
            else:
                return False

    def _whichChild(self, node):
        flag = None
        parent = node.parent
        children = parent.getChildNodes()
        if children[0] is node:
            flag = 1
        elif children[1] is node:
            flag = 2
        return flag

    def resetAllPruneNodes(self, node):
        if node.isPrune() or node.includedInCluster:
            node.setPruneState(False)
            node.includedInCluster = False
            node.clusterId = None
            node.touchedNodes = None
            node.modifiedLength = None

        children = node.getChildNodes()
        if len(children) == 0:
            return

        elif len(children) == 1:
            self.resetAllPruneNodes(children[0])

        else:
            self.resetAllPruneNodes(children[0])
            self.resetAllPruneNodes(children[1])

    def _calcClusterMemebersVariance(self, totalDataInstancesbyEachGroup):
        if len(totalDataInstancesbyEachGroup)< 2:
            logger.warning("Couldn't divide the data into clusters with current setting!")
            return float('inf'), float('inf')
        vals = list(totalDataInstancesbyEachGroup.values())
        mean = np.mean(vals)
        var = np.var(vals)
        return mean, var

    def clusterStatistics(self, originalRepresentationVectors, finalRepresentationVectors,
                          finalRepresentationVectorsToCluster, totalDataInstancesbyEachGroup):
        if not self.useSilhouette:
            if self.balancedPrune:
                qualityMetric = 'varPurity'
            else:
                qualityMetric = 'purity'
        else:
            qualityMetric = 'silhouette'

        originalRepresentationVectors = self._getUnitVectorOfRepresentationVectors(originalRepresentationVectors)
        finalRepresentationVectors = self._getUnitVectorOfRepresentationVectors(finalRepresentationVectors)
        result = {}
        totalDataPointsByClusters = {}
        finalToOriginalRepresentationVectorsMap = {}

        for representationVectorKey_F in finalRepresentationVectors.keys():
            l = []
            for representationVectorKey_O in originalRepresentationVectors.keys():
                if len(set(representationVectorKey_O) - set(representationVectorKey_F)) == 0:
                    l.append(representationVectorKey_O)
            finalToOriginalRepresentationVectorsMap[representationVectorKey_F] = l
            cluster = finalRepresentationVectorsToCluster[representationVectorKey_F]
            totalDataPoints = totalDataInstancesbyEachGroup[representationVectorKey_F]
            totalDataPointsByClusters[cluster] = totalDataPoints
        if not self.useSilhouette:
            allCombosOfFinalRepresentationVectors =\
                sum([list(map(list, combinations(list(finalRepresentationVectors.keys()), i))) for i in range(2, 3)], [])
            # interClusterDistance = Avg. distance between clusters. More the better.
            # Because you want to maintain as much heterogeneity among different clusters.
            # intraClusterDistance = Avg. distance within clusters. Less the better.
            # Because you want maintain as much homogeneity as possible in a cluster.
            interClusterDistance, intraClusterDistance = self._calcClusterStatistics(allCombosOfFinalRepresentationVectors,
                                                                                    finalRepresentationVectors,
                                                                                    originalRepresentationVectors,
                                                                                    finalToOriginalRepresentationVectorsMap)
            # purity = Actually inverse purity, Less the better. Formula: intraClusterDistance/interClusterDistance
            purity = intraClusterDistance / interClusterDistance
            mean, var = self._calcClusterMemebersVariance(totalDataPointsByClusters)
            varPurity = np.sqrt(var) ** purity # We want a balanced cluster at the same time a homogeneous cluster if balancedPrune
            result = {'intra-cluster-distance': intraClusterDistance,
                      'inter-cluster-distance': interClusterDistance,
                      'purity': purity,
                      'varPurity': varPurity,
                      'mean-members': mean,
                      'data-points': totalDataPointsByClusters
                     }
            result['argminMetric'] = result[qualityMetric]
        else:
            individualPairWiseFinalRepresentationVectors = {}
            finalRepresentationVectorsKeys = list(finalRepresentationVectors.keys())
            for i in finalRepresentationVectorsKeys:
                pairs = []
                finalRepresentationVectorsKeysCopy = finalRepresentationVectorsKeys.copy()
                finalRepresentationVectorsKeysCopy.remove(i)
                pairs = [[i, j] for j in finalRepresentationVectorsKeysCopy]
                individualPairWiseFinalRepresentationVectors[i] = pairs

            silhouetteConstants = self._calcSilhouetteConstants(
                individualPairWiseFinalRepresentationVectors,
                finalRepresentationVectors, originalRepresentationVectors,
                finalToOriginalRepresentationVectorsMap)

            silhouetteIndicator = np.mean(list(silhouetteConstants.values()))

            result = {'silhouetteConstants': silhouetteConstants,
                      'silhouette': silhouetteIndicator,
                      'data-points': totalDataPointsByClusters}

        return result

    def _calcSilhouetteConstants(self, individualPairWiseFinalRepresentationVectors, finalRepresentationVectors,
                                 originalRepresentationVectors, finalToOriginalRepresentationVectorsMap):

        individualIntraClusterDistance = {}
        for finalCluster in finalToOriginalRepresentationVectorsMap.keys():
            finalClusterRepresentationVectors = finalRepresentationVectors[finalCluster]
            dist = 0.
            for origGroup in finalToOriginalRepresentationVectorsMap[finalCluster]:
                origClusterRepresentationVectors = originalRepresentationVectors[origGroup]
                dist += self._calcEuclidianDistance(finalClusterRepresentationVectors, origClusterRepresentationVectors)

            dist = dist / len(finalToOriginalRepresentationVectorsMap[finalCluster])
            individualIntraClusterDistance[finalCluster] = dist

        individualInterClusterDistance = {}
        for finalCluster in individualPairWiseFinalRepresentationVectors.keys():
            interClusterDistance = 0.
            pairs = individualPairWiseFinalRepresentationVectors[finalCluster]
            for pair in pairs:
                fc1 = finalRepresentationVectors[pair[0]]
                fc2 = finalRepresentationVectors[pair[1]]
                dist = self._calcEuclidianDistance(fc1, fc2)
                interClusterDistance += dist

            interClusterDistance = interClusterDistance / len(pairs)
            individualInterClusterDistance[finalCluster] = interClusterDistance

        silhouetteConstants = {}
        for finalCluster in finalRepresentationVectors.keys():
            silhouetteConstant = \
                (individualInterClusterDistance[finalCluster] - individualIntraClusterDistance[finalCluster]) / \
                np.maximum(individualInterClusterDistance[finalCluster], individualIntraClusterDistance[finalCluster])

            silhouetteConstants[finalCluster] = silhouetteConstant

        return silhouetteConstants

    def _calcClusterStatistics(self, allCombosOfFinalRepresentationVectors, finalRepresentationVectors,
                               originalRepresentationVectors, finalToOriginalRepresentationVectorsMap):

        interClusterDistance = 0.  # Avg. distance between clusters. More the better.
        # Because you want to maintain as much heterogeneity among different clusters.
        intraClusterDistance = 0.  # Avg. distance within clusters. Less the better.
        # Because you want maintain as much homogeneity as possible in a cluster.
        for combos in allCombosOfFinalRepresentationVectors:
            fc1 = finalRepresentationVectors[combos[0]]
            fc2 = finalRepresentationVectors[combos[1]]
            dist = self._calcEuclidianDistance(fc1, fc2)
            interClusterDistance += dist
        interClusterDistance = interClusterDistance / len(finalRepresentationVectors.keys())

        individualIntraClusterDistance = {}
        for finalCluster in finalToOriginalRepresentationVectorsMap.keys():
            finalClusterRepresentationVectors = finalRepresentationVectors[finalCluster]
            dist = 0.
            for origGroup in finalToOriginalRepresentationVectorsMap[finalCluster]:
                origClusterRepresentationVectors = originalRepresentationVectors[origGroup]
                dist += self._calcEuclidianDistance(finalClusterRepresentationVectors, origClusterRepresentationVectors)

            dist = dist / len(finalToOriginalRepresentationVectorsMap[finalCluster])
            individualIntraClusterDistance[finalCluster] = dist

        intraClusterDistance = sum(individualIntraClusterDistance.values()) / len(finalRepresentationVectors.keys())
        return interClusterDistance, intraClusterDistance

    def _calcEuclidianDistance(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def getClusterID(self):
        self.maxSoFar += 1
        return self.prefixString + '_' + "cluster_" + str(self.maxSoFar)

    def pruningRedundantNodes(self, root):
        preOrderedListofClusters = self.preOrderTraversalofClusters(root, [])
        preOrderedListofClusters = [node for _, _, node in preOrderedListofClusters]
        if len(preOrderedListofClusters) < 3:
            return

        curPosition = 0
        nextPosition = 1
        while True:
            if nextPosition <= curPosition or nextPosition >= len(preOrderedListofClusters):
                break

            curNode = preOrderedListofClusters[curPosition]
            nxtNode = preOrderedListofClusters[nextPosition]
            curNodeParent = curNode.getParent()
            nxtNodeParent = nxtNode.getParent()
            curNodeWhichChild = self._whichChild(curNode)
            nxtNodeWhichChild = self._whichChild(nxtNode)
            if curNodeWhichChild != nxtNodeWhichChild or curNode.clusterId != nxtNode.clusterId or\
                    curNode.parent.attribute != nxtNode.parent.attribute:
                curPosition += 1
                nextPosition += 1
                continue
            else:
                if curNodeWhichChild == 1:
                    # Both are left child of their respective parents.
                    nxtNodeGrandParent = nxtNode.getParent().getParent()
                    if curNodeParent == nxtNodeGrandParent and \
                            curNodeParent is not None and nxtNodeGrandParent is not None:
                        # Replacing curNodeParent and curNode by
                        # curNodeParent_RightChild(sibling of curNode) and
                        # curNodeParent_RightChild_LeftChild(nxtNode) respectively.
                        curNodeGrandParent = curNodeParent.getParent()
                        if curNodeGrandParent is not None:
                            curNodeParentWhichChild = self._whichChild(curNodeParent)
                            curNodeGrandParent.children[curNodeParentWhichChild - 1] = nxtNodeParent
                        else:
                            self.root = nxtNodeParent
                            self.cltree.root = nxtNodeParent
                            allNodes = self.cltree._collectAllNodes(self.cltree.root, dict())
                            self.cltree.nodes = allNodes

                        nxtNodeParent.parent = curNodeGrandParent
                        nxtNodeParent.dataset = curNodeParent.dataset
                        nxtNodeDataSet = self._merging2Datasets(curNode.dataset, nxtNode.dataset)
                        nxtNode.dataset = nxtNodeDataSet
                        preOrderedListofClusters.remove(curNode)

                    else:
                        curPosition += 1
                        nextPosition += 1

                elif curNodeWhichChild == 2:
                    # Both are right child of their respective parents.
                    curNodeGrandParent = curNodeParent.getParent()
                    if curNodeGrandParent == nxtNodeParent and \
                            nxtNodeParent is not None and curNodeGrandParent is not None:
                        # Replacing curNodeGrandParent/nxtNodeParent by curParent
                        nxtNodeGrandParent = nxtNodeParent.getParent()
                        if nxtNodeGrandParent is not None:
                            nxtNodeParentWhichChild = self._whichChild(nxtNodeParent)
                            nxtNodeGrandParent.children[nxtNodeParentWhichChild - 1] = curNodeParent
                        else:
                            self.root = curNodeParent
                            self.cltree.root = curNodeParent
                            allNodes = self.cltree._collectAllNodes(self.cltree.root, dict())
                            self.cltree.nodes = allNodes

                        curNodeParent.parent = nxtNodeGrandParent
                        curNodeParent.dataset = nxtNodeParent.dataset
                        curNodeDataSet = self._merging2Datasets(curNode.dataset, nxtNode.dataset)
                        curNode.dataset = curNodeDataSet
                        preOrderedListofClusters.remove(nxtNode)

                    else:
                        curPosition += 1
                        nextPosition += 1

        self._recalculateDepthOfEachNodes(self.cltree.root, 0)
        return

    def _recalculateDepthOfEachNodes(self, node, depth):
        node.depth = depth
        childrens = node.getChildNodes()
        if childrens is not None and len(childrens) > 0:
            self._recalculateDepthOfEachNodes(childrens[0], depth + 1)
        if childrens is not None and len(childrens) > 1:
            self._recalculateDepthOfEachNodes(childrens[1], depth + 1)
        return

    def _merging2Datasets(self, dataset1, dataset2):
        assert isinstance(dataset1, Data) and isinstance(dataset2, Data), \
            "_merging2Datasets() can handle only instance of ADT 'Data' class!"
        dataset1_instance_values = dataset1.instance_values
        dataset1_attr_types = dataset1.attr_types
        dataset1_attr_idx = dataset1.attr_idx

        dataset2_instance_values = dataset2.instance_values
        dataset2_attr_types = dataset2.attr_types
        dataset2_attr_idx = dataset2.attr_idx

        if len(dataset1_attr_idx) != len(dataset2_attr_idx) and \
                len(set(dataset1_attr_idx.keys()) - set(dataset2_attr_idx.keys())) != 0:
            logger.error('Could not merge 2 datasets due to datatype/column mismatch.\
                                    dataset1 attr. types: {}, dataset2 attr. types: {}'.format(dataset1_attr_types,
                                                                                   dataset2_attr_types))
            raise ValueError('Could not merge 2 datasets due to datatype/column mismatch.\
                                    dataset1 attr. types: {}, dataset2 attr. types: {}'.format(dataset1_attr_types,
                                                                                   dataset2_attr_types))
        else:
            for k in dataset1_attr_idx.keys():
                dataset1_k_idx = dataset1_attr_idx[k]
                dataset2_k_idx = dataset2_attr_idx[k]
                if dataset1_k_idx != dataset2_k_idx:
                    temp = np.copy(dataset2_instance_values[:, dataset1_k_idx])
                    dataset2_instance_values[:, dataset1_k_idx] = dataset2_instance_values[:, dataset2_k_idx]
                    dataset2_instance_values[:, dataset2_k_idx] = temp

            newDataSetInstances = np.concatenate((dataset1_instance_values, dataset2_instance_values), axis=0)
            dataset = Data(newDataSetInstances, dataset1_attr_types)
            return dataset


def pruneByGridSearch_Centroid(cltree, min_y, data_length, prefixString, balancedPrune, useSilhouette):
    assert isinstance(cltree, CLTree) and isinstance(cltree.getRoot(), CLNode), logger.error(\
               "ERROR! Data Type mistmatch. This function can operate only on CLTree and CLNode")
    root = cltree.getRoot()
    gridSearch = {}
    validateResults = {}

    # Validating the value of min_y
#    if option == 2:
#         if not (min_y >= data_length*0.07 and min_y <= data_length*0.15):
#             min_y = data_length*0.10
#             logger.warning("The 'min_y' go modified to {}!".format(min_y))
#    else:
#         if not (min_y >= data_length*0.10 and min_y <= data_length*0.20):
#             min_y = data_length*0.15
#             logger.warning("The 'min_y' got modified to {}!".format(min_y))

    minLowerLimit = min_y - int(min_y*0.1)

    for min_members in list(np.ceil(np.linspace(start=minLowerLimit, stop=min_y, num=10))):
        p = PruneTreeByMergingCentroids(prefixString, cltree, balancedPrune, useSilhouette)
        p.resetAllPruneNodes(node=root)
        result = p.prune(min_y=min_members, searchMode=True)
        # Result will be None, if algorithm couldn't find meaningful clusters each containing atleast min_members.
        if result is not None:
            validateResults[min_members] = result
            if not useSilhouette:
                argminMetric = result['argminMetric']
                gridSearch[min_members] = argminMetric
            else:
                silhouette = result['silhouette']
                gridSearch[min_members] = silhouette


    if len(gridSearch) == 0:
        logger.error("ATTENTION: No meaningful clusters found with atleast {} data points"
                     " with balancedPrune = {}.".format(min_y, balancedPrune))
        cltree.root.setPruneState(True)
        cltree.root.includedInCluster = True
        cltree.root.clusterId = prefixString + '_'+ 'DEFAULT'
        result = {'intra-cluster-distance': None,
                  'inter-cluster-distance': None,
                  'purity': None,
                  'silhouette': None,
                  'mean-members': data_length,
                  'data-points': data_length
                 }
        cltree.setClustersStatistics(result)
        baseVer = p.version
        return result, baseVer
        #raise RuntimeError("No meaningful clusters found with atleast {} data points.".format(min_y))

    else:
        logger.info("balancedPrune: {}, All argminMetric: {}".format(balancedPrune, gridSearch))
        logger.info("All intermediate stats, generated during grid search: {}".format(validateResults))
        p = PruneTreeByMergingCentroids(prefixString, cltree, balancedPrune, useSilhouette)
        p.resetAllPruneNodes(node=root)
        if not useSilhouette:
            finalMinY = min(gridSearch, key=gridSearch.get)
        else:
            finalMinY = max(gridSearch, key=gridSearch.get)

        result = p.prune(min_y=finalMinY, searchMode=False)
        cltree.setClustersStatistics(result)
        baseVer = p.version
        return result, baseVer
