#pragma once
#include <vector>
#include "../Util/Types.hpp"

namespace LocalDescriptorAndBagOfFeature 
{    
    class GMM
    {
        public:
            GMM(int numGaussians);
            
            void Train(const FeatureSet &featureSet);
            std::vector<double> Supervector(const BagOfFeatures &bof);
            
            int NumGaussians(void) const;
            
        private:
            int _NumGaussians;
    };
}
