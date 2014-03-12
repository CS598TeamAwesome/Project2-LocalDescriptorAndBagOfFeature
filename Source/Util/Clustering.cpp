#include "Clustering.hpp"

struct bin_info {
    int size;
    std::vector<double> sum;
    std::vector<double> mean;
};

void vector_add(std::vector<double> &v1, std::vector<double> &v2){
    assert(v1.size() == v2.size());

    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::plus<double>());
}

void vector_subtract(std::vector<double> &v1, std::vector<double> &v2){
    assert(v1.size() == v2.size());

    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), std::minus<double>());
}

double euclidean_distance(const std::vector<double> &v1, const std::vector<double> &v2){
    assert(v1.size() == v2.size());

    //get differences
    std::vector<double> diff(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), diff.begin(), std::minus<double>());

    //sum squares and take sqrt
    double sum2 = std::accumulate(diff.begin(), diff.end(), 0.0, [](double accum, double elem) { return accum + elem * elem; });
    double sum = std::sqrt(sum2);

    return sum;
}


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
double LocalDescriptorAndBagOfFeature::kmeans(std::vector<std::vector<double>> input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes){
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
    while(recompute){
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

        //4. recompute mean for new centers
        for(bin_info& binfo : totals){
            for(int i = 0; i < dim; i++){
                binfo.mean[i] = binfo.sum[i]/binfo.size;
            }
        }
    }

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
double LocalDescriptorAndBagOfFeature::kmeans(const std::vector<std::vector<double>> &input, int K, std::vector<int> &labels, std::vector<std::vector<double>> &centers, std::vector<int> &sizes, int trials){
    //initialize best results
    std::vector<std::vector<double>> best_centers;
    std::vector<int> best_labels;
    std::vector<int> best_sizes;
    double best_compactness = kmeans(input, K, best_labels, best_centers, best_sizes); //first trial

    for(int i = 1; i < trials; i++){
        std::vector<std::vector<double>> current_centers;
        std::vector<int> current_labels;
        std::vector<int> current_sizes;
        double current_compactness = kmeans(input, K, current_labels, current_centers, current_sizes);

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
