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
 * @param K -- the number of clusters to divide them into
 * @return a vector of the K cluster centers
 *
 * Bare bones implementation.
 * - Assigns initial distribution using mod on index.
 * - Runs until local minimum is reached.
 * - Uses Euclidean distance
 */
std::vector<std::vector<double>> LocalDescriptorAndBagOfFeature::kmeans(std::vector<std::vector<double>> &input, int K){
    int sample_ct = input.size();

    //1. assign vectors to bins
    std::vector<int> current_bins(input.size());
    for(int i = 0; i < sample_ct; i++){
        current_bins[i] = i%K; //initial bin assignments, fixed strategy, completely arbitrary, very easy to code
    }

    //x. initial setup for computing mean vector for each bin
    std::vector<bin_info> totals(K);

    int dim = input[0].size(); //dimension of samples, maybe cleaner to pass this in

    //initialize bin size, sum, and mean
    for(bin_info& binfo : totals){
        binfo.size = 0;
        binfo.sum.resize(dim);
        binfo.mean.resize(dim);
    }

    //determine totals for initial distribution
    for(int i = 0; i < sample_ct; i++){
        int bin_index = current_bins[i];

        //add the vector to the total for the bin
        vector_add(totals[bin_index].sum, input[i]);
        totals[bin_index].size++;
    }

    bool recompute = true;
    int iteration_ct = 0;
    while(recompute){
        iteration_ct++;

        //2. compute bin means
        for(bin_info& binfo : totals){
            for(int i = 0; i < dim; i++){
                binfo.mean[i] = binfo.sum[i]/binfo.size;
            }
        }

        //3. compare each vector to each bin mean and note most similar (euclidean distance)
        std::vector<int> new_bins(sample_ct);

        for(int i = 0; i < sample_ct; i++){

            int nearest_bin = 0;
            double nearest_distance = euclidean_distance(input[0], totals[0].mean);

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

        recompute = false;
        //4. move all vectors to most similar bin
        for(int i = 0; i < sample_ct; i++){
            if(new_bins[i] != current_bins[i]){
                //remove vector from current bin
                totals[current_bins[i]].size--;
                vector_subtract(totals[current_bins[i]].sum, input[i]);

                //put vector into new bin
                totals[new_bins[i]].size++;
                vector_add(totals[new_bins[i]].sum, input[i]);
                current_bins[i] = new_bins[i];

                recompute = true; //state changed, so run another iteration
            }
        }
    }

    //5. local minimum reached, return a list of cluster centers
    std::vector<std::vector<double>> centers;

    //pull the means out from the vector of bin_infos
    for(bin_info& binfo : totals){
        centers.push_back(binfo.mean);
    }

    return centers;
}
