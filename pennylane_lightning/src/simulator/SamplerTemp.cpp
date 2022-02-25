#include <iostream>
#include <vector>
#include <cstddef>
#include <numeric>
#include <random>
#include <cassert>
#include <chrono>

struct WalkerData {

  int N;
  std::vector<double> bucket_share;
  std::vector<int> bucket_partner;

  //probs must be normalized and they are edited in place
  WalkerData(std::vector<double> & probs){

    N = probs.size();
    bucket_share = std::vector<double>(N);
    bucket_partner = std::vector<int>(N);
    
    double sum = 0;

    for (int i=0; i<N; i++) { sum += probs[i]; }
    for (int i=0; i<N; i++) {
      bucket_share[i] = probs[i] / (sum/N); 
      bucket_partner[i] = i; 
    }

    int i,j,k;
    //Find first overpopulated bucket
    for (j=0; j<N && !(bucket_share[j] > 1.0); j++) {/*seek*/}

    for (i=0; i<N; i++) {
      k = i; // k is bucket under consideration
      if (bucket_partner[k]!=i) {
	continue;  // reject already considered buckets
      }

      double excess = 1.0 - bucket_share[k];  
      while (excess > 0 ) {
	if (j == N) { break; }     
	bucket_partner[k]=j;              
	bucket_share[j] -= excess;
	excess = 1.0 - bucket_share[j];  
	if (excess >= 0) { 
	  for (k=j++; j<N && !(bucket_share[j] > 1.0); j++) {/*seek*/}
	}
      }
    }

  
  }
};

std::vector<int> SimpleSample(const std::vector<double> & probs, int num_samples)
{
  int N = probs.size();
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0,1.0);
  std::vector<int> counts(N,0);
    
  for (int i = 0; i < num_samples; i++){
    int num_elements = probs.size();
    int pick = 0;
    double dice = dist(mt);
    double checkpoint = 0;
    while (pick < num_elements){
      checkpoint += probs[pick];
      // std::cout << "dice " << dice << std::endl; 
      // std::cout << "checkpoint " << checkpoint << std::endl;
      if (dice < checkpoint) break;
      pick++;
    }
    counts[pick] += 1;
  }
  
  return counts;
}

double doubleRand(double min, double max) {
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}

std::vector<int> WalkerSample(const WalkerData & wd, int num_samples){

  int N = wd.N;
  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_real_distribution<double> dist(0.0,1.0);
  std::vector<int> counts(N,0);

  #pragma omp parallel for 
  for (int i = 0; i < num_samples; i++){
    // double pct = dist(mt)*N;
    double pct = doubleRand(0.,1.)*N;
    int idx = static_cast<int>(pct);
    if (pct-idx > wd.bucket_share[idx])
      counts[wd.bucket_partner[idx]] += 1;
    else
      counts[idx] += 1;
  }

  return counts;
}

std::vector<double> GetProbabilityDist(int N){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0.0,1.0);
  std::vector<double> probs(N);
  for (int i = 0; i < N; i++){
    probs[i] = dist(mt);
    // std::cout << probs[i] << std::endl;
  }
  double norm = std::accumulate(probs.begin(), probs.end(), 0.0);
  // std::cout << norm << std::endl;
  
  for (int i = 0; i < N; i++){
    probs[i] /= norm;
  }
  return probs;
}

std::vector<double> EstimateProbabilityFromCounts(std::vector<int> & counts, int num_samples){
  std::vector<double> probs(counts.size());
  for (int i = 0; i < counts.size(); i++){ 
    probs[i] = counts[i]/(double)num_samples;
  }
  return probs;
}

template <typename T>
void PrintVector(std::vector<T> & vec){
  
  for (int i = 0; i < vec.size(); i++)
    std::cout << vec[i] << " , ";
  std::cout << std::endl;

}

auto MaxError(
	      std::vector<double> & v1,
	      std::vector<double> & v2
	     )
{
  assert(v1.size() == v2.size());
  double max_error = -1;
  for (size_t i = 0; i < v1.size(); i++){
    double error = fabs(v1[i]-v2[i]);
    if (error > max_error){
      max_error = error;
    }
  }
  return max_error;
}


int main(int argc, char *argv[])
{
  if (argc != 3){
    std::cout << "sampler <NUM_ELEMENTS> <NUM_SAMPLES>" << std::endl;
    exit(1);
  }
  
  int num_elements = std::stoi(argv[1]);
  int num_samples = std::stoi(argv[2]);
  auto probs = GetProbabilityDist(num_elements);

  // auto start = std::chrono::high_resolution_clock::now();
  // auto counts_simple = SimpleSample(probs, num_samples);
  // auto stop = std::chrono::high_resolution_clock::now();
  // auto simple_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
 
  auto start = std::chrono::high_resolution_clock::now();  
  WalkerData wd(probs);
  auto counts_walker = WalkerSample(wd, num_samples);
  auto stop = std::chrono::high_resolution_clock::now();
  auto walker_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  
  // auto counts_simple_prob_est = EstimateProbabilityFromCounts(counts_simple,num_samples);
  auto counts_walker_prob_est = EstimateProbabilityFromCounts(counts_walker,num_samples);

  // PrintVector(probs);
  // PrintVector(counts_simple);
  // PrintVector(counts_simple_prob_est);

  // auto err = MaxError(counts_simple_prob_est,probs);
  auto w_err = MaxError(counts_walker_prob_est,probs);

  // std::cout << "err = " << err << " " << .001*simple_duration.count() << std::endl;
  std::cout << "w_err = " << w_err << " " << .001*walker_duration.count() << std::endl;
  // for (int i = 0; i < num_elements; i++)
  //   std::cout << counts_simple_prob_est[i] << " , ";
  // std::cout << std::endl;

  // for (int i = 0; i < num_elements; i++)
  //   std::cout << counts_walker_prob_est[i] << " , ";
  // std::cout << std::endl;
  
  return 0;
}

