#include "Clustering.hpp"

struct bin_info {
    int size;
    std::vector<double> sum;
    std::vector<double> mean;
    std::vector<double> old_mean;
};

/**
 * @brief LocalDescriptorAndBagOfFeature::kmeans - computes K cluster centers for given samples
 * @param input -- the samples, a vector of vector double -- assumed to be of equal length
 *      ... this is pass by value, because we rearrange elements pseudorandomly in generating the initial seeding
 * @param K -- the number of clusters to divide them into
 * @param labels -- the bin labels for each sample
 * @param centers -- the mean vectors for each cluster
 * @return a vector of the K cluster centers, a vector of bin labels each sample belongs in, the compactness score for the clustering
 *
 * Bare bones implementation.
 * - Assigns initial distribution using mod on index.
 * - Runs until local minimum is reached.
 * - Uses Euclidean distance
 */
double LocalDescriptorAndBagOfFeature::kmeans(std::vector<std::vector<double>> input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int iteration_bound, int epsilon){
    int sample_ct = input.size();
    int dim = input[0].size(); //dimension of samples, maybe cleaner to pass this in

    //1.initial seeding of cluster centers
    std::vector<bin_info> totals(K);

    for(bin_info& binfo : totals){
        binfo.size = 0;
        binfo.sum.resize(dim);
        binfo.mean.resize(dim);
    }

    //pick K random samples to act as centers
    for(int i = 0; i < K; i++){
        //pick random sample in window from i to end
        int index = i + std::rand()%(sample_ct-i);

        //assign the sample as a cluster center
        std::vector<double> v(input[index]);
        totals[i].mean = v;

        //swap chosen sample into the current index to avoid collision (choosing the same sample twice)
        std::swap(input[index], input[i]);
    }

    std::vector<int> current_bins(input.size());
    for(int i = 0; i < sample_ct; i++){
        current_bins[i] = -1;
    }

    bool recompute = true;
    int iteration_ct = 0;
    while(recompute && iteration_ct < iteration_bound){
        //print out iteration count for larger set sizes
        if(input.size() > 25000){
            std::cout << input.size() << " samples... iteration: " << iteration_ct << std::endl;
        }
        iteration_ct++;

        //2. compare each sample to each bin mean and note most similar (euclidean distance)
        std::vector<int> new_bins(sample_ct);
        for(int i = 0; i < sample_ct; i++){
            int nearest_bin = 0;
            double nearest_distance = euclidean_distance(input[i], totals[0].mean);

            for(int j = 0; j < K; j++){
                double distance = euclidean_distance(input[i], totals[j].mean);

                //keep track of nearest bin
                if(distance < nearest_distance){ //strict less than means in the case of a tie, we pick the lowest index
                    nearest_bin = j;
                    nearest_distance = distance;
                }
            }

            new_bins[i] = nearest_bin;
        }

        //3. move each sample to bin with closest center
        recompute = false;
        for(int i = 0; i < sample_ct; i++){
            if(new_bins[i] != current_bins[i]){
                if(current_bins[i] != -1){
                    //remove vector from current bin
                    totals[current_bins[i]].size--;
                    vector_subtract(totals[current_bins[i]].sum, input[i]);
                }

                //put vector into new bin
                totals[new_bins[i]].size++;
                vector_add(totals[new_bins[i]].sum, input[i]);
                current_bins[i] = new_bins[i];

                recompute = true; //state changed, so run another iteration
            }
        }

        //4. recompute mean for new centers -- keeping track of how much it moved
        double max_move = 0;
        for(bin_info& binfo : totals){
            if(binfo.size == 0){
                std::cout << "A bin is empty... re-assign random sample to it" << std::endl;
                int index = std::rand()%(sample_ct);
                std::vector<double> v(input[index]);
                binfo.mean = v;
            } else {
                double sum = 0.0;
                for(int i = 0; i < dim; i++){
                    double old_value = binfo.mean[i];
                    double new_value = binfo.sum[i]/binfo.size;
                    binfo.mean[i] = new_value;

                    sum += ((old_value - new_value)*(old_value - new_value));
                }
                if(sum > max_move)
                    max_move = sum;
            }
        }

        if(input.size() > 25000)
            std::cout << "max center shift: " << max_move << std::endl;

        //termination condition: no center moved more than epsilon distance, so approaching local minimum
        if(max_move < epsilon){
            recompute = false;
        }
    }

    //std::cout << "kmeans ran for: " << iteration_ct << " iterations" << std::endl;
    //5. local minimum reached

    //set cluster centers: pull the means out from the vector of bin_infos
    centers.clear();
    for(bin_info& binfo : totals){
        centers.push_back(binfo.mean);
    }

    //set cluster sizes: from binfo
    sizes.clear();
    for(bin_info& binfo : totals){
        sizes.push_back(binfo.size);
    }

    //set labels
    labels.clear();
    for(int& i: current_bins){
        labels.push_back(i);
    }

    //6. compute and return compactness:
    // -- which we will define as average euclidean distance between each sample and the center of its cluster
    double sum = 0.0;
    for(int i = 0; i < sample_ct; i++){
        double distance = euclidean_distance(input[i], centers[labels[i]]);
        sum += distance*distance; //unsure if squaring here is necessary
    }
    //some measures also divide the sum by number of samples
    sum /= sample_ct;

    return sum;
}

/**
 * @brief LocalDescriptorAndBagOfFeature::kmeans
 *  -- run kmeans for N trials and return the best one
 */
double LocalDescriptorAndBagOfFeature::kmeans(const std::vector<std::vector<double>> &input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int iteration_bound, int epsilon, int trials){
    //initialize best results
    std::vector<std::vector<double>> best_centers;
    std::vector<int> best_labels;
    std::vector<int> best_sizes;
    double best_compactness = kmeans(input, K, best_labels, best_centers, best_sizes, iteration_bound, epsilon); //first trial

    for(int i = 1; i < trials; i++){
        if(input.size() > 25000)
            std::cout << "k-means trial#: " << i << std::endl;
        std::vector<std::vector<double>> current_centers;
        std::vector<int> current_labels;
        std::vector<int> current_sizes;
        double current_compactness = kmeans(input, K, current_labels, current_centers, current_sizes, iteration_bound, epsilon);
        if(input.size() > 25000)
            std::cout << ".. current compactness: " << current_compactness << std::endl;

        if(current_compactness < best_compactness){
            best_compactness = current_compactness;
            best_centers = current_centers;
            best_labels = current_labels;
            best_sizes = current_sizes;
        }
    }

    //set cluster centers: pull the means out from the vector of bin_infos
    centers.clear();
    for(std::vector<double>& bin_mean : best_centers){
        centers.push_back(bin_mean);
    }

    //set cluster sizes
    sizes.clear();
    for(int& i: best_sizes){
        sizes.push_back(i);
    }

    //set labels
    labels.clear();
    for(int& i: best_labels){
        labels.push_back(i);
    }    

    return best_compactness;
}

//the tree's K and L should be set prior to call, the tree will then be populated by the algorithm
void LocalDescriptorAndBagOfFeature::hierarchical_kmeans(const std::vector<std::vector<double>> &input, vocabulary_tree &tree){
    hierarchical_kmeans(input, tree.K, tree.L, tree.root);
}

void LocalDescriptorAndBagOfFeature::hierarchical_kmeans(const std::vector<std::vector<double>> &input, int K, int L, tree_node &root){
    //base case no more levels
    if(L == 0){
        return;
    }

    //base case not enough children
    if(input.size() <= K){
        for(int i = 0; i < input.size(); i++){
            std::cout << L << ", " << i << ", " << K << "no children" << std::endl;
            tree_node child;
            std::vector<double> v(input[i]);
            child.value = v;
        }
        return;
    }

    std::vector<std::vector<double>> centers;
    std::vector<int> labels;
    std::vector<int> sizes;
    //iteration cap 15, epsilon 100, trials 1
    kmeans(input, K, labels, centers, sizes, 15, 100, 1);
    root.children.clear();
    for(int i = 0; i < K; i++){
        std::vector<std::vector<double>> cluster;
        //build sub-cluster to pass into sub-tree
        for(int j = 0; j < input.size(); j++){
            if(labels[j] == i){
                cluster.push_back(input[j]);
            }
        }

        tree_node child;
        child.value = centers[i];
        hierarchical_kmeans(cluster, K, L-1, child);
        root.children.push_back(child);
    }
}
