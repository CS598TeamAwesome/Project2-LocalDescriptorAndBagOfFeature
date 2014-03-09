#include <vector>

namespace LocalDescriptorAndBagOfFeature 
{
    class GMM
    {
        public:
            GMM(int numGaussians);
            
            void Train(const std::vector<std::vector<std::vector<double>>> &bof);
            std::vector<double> Supervector(const std::vector<std::vector<double>> &bof);
            
            int NumGaussians(void) const;
            
        private:
            int _NumGaussians;
    };
}
