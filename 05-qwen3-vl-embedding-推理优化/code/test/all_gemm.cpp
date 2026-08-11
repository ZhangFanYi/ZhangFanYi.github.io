#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <assert.h>

#include <omp.h>
#include <random>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#define WARMUP
#include <sys/time.h>
#include <chrono>

using namespace std;
// Timer struct using chrono for high resolution timing
struct my_timer {
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    double time_use; // us
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        time_use = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }
};

double get_time_us_sync(hipStream_t stream) {
    hipStreamSynchronize(stream);
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<double>(duration);
}

rocblas_operation getTranspose(char trans){
    if(trans == 'N' || trans == 'n')
        return rocblas_operation_none;           
    else if(trans == 'T' || trans == 't')
        return rocblas_operation_transpose;
	else 
		return rocblas_operation_conjugate_transpose;
}
void updateProgressBar(int progress, int total) {
    int barWidth = 70;
    std::cout << "RUN[";
    int pos = barWidth * progress / total;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress / (total / 100.0)) << " %\r";
    std::cout.flush();
}
void handleHipError(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        std::cerr << "HIP error at " << file << ":" << line << " code=" << err << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void handleRocblasError(rocblas_status err, const char* file, int line) {
    if (err != rocblas_status_success) {
        std::cerr << "rocBLAS error at " << file << ":" << line << " code=" << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(err) (handleHipError(err, __FILE__, __LINE__))
#define CHECK_ROCBLAS_ERROR(err) (handleRocblasError(err, __FILE__, __LINE__))
size_t freeMem = 0;
size_t totalMem = 0;
size_t failed_size_count = 0;


template <typename Ti,typename To, typename Tc>
int gemm_init_gemm(char** argv, string gemmType)
{
	ifstream inFile(argv[1]);
    if (!inFile.is_open()) {
		cerr << "Can't open the input file!" << endl;
        return EXIT_FAILURE;
	}

	int warmup_num = atoi(argv[2]);
	int iter_num = atoi(argv[3]);

	size_t max_m = 0;
	size_t max_n = 0;
	size_t max_k = 0;
	size_t max_batchCount = 0;
	size_t count = 0;

	std::vector<size_t> vec_m;
	std::vector<size_t> vec_n;
	std::vector<size_t> vec_k;
	std::vector<size_t> vec_batchcnt;
	std::vector<char> vec_trans_a;
	std::vector<char> vec_trans_b;

	std::vector<double> vec_gflops_results;
	std::vector<double> vec_time_costs_results;

	string line;
	while(getline(inFile, line))
	{
		// read input parameter
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;
		stringstream ss(line);
		ss >> trans_a;
		ss >> trans_b;
		ss >> m;
		ss >> n;
		ss >> k;
		ss >> batchCount;
		vec_trans_a.push_back(trans_a);
		vec_trans_b.push_back(trans_b);	
		vec_m.push_back(m);
		vec_n.push_back(n);
		vec_k.push_back(k);
		vec_batchcnt.push_back(batchCount);
																									 			
        max_m = max(max_m, m);
        max_n = max(max_n, n);
        max_k = max(max_k, k);
        max_batchCount = max(max_batchCount, batchCount);

		count++;
	}
    cout << "Total lines: " << count << ", max_m: " << max_m << ", max_n: " << max_n << ", max_k: " << max_k << ", max_batchCount: " << max_batchCount << endl;
	int num_count = count;
	inFile.close();

	std::vector<long long> vec_mn;
	std::vector<long long> vec_mk;
	std::vector<long long> vec_nk;	
	for(int idx = 0; idx < num_count; idx++)
	{
		vec_mn.push_back(vec_m[idx]*vec_n[idx]*vec_batchcnt[idx]);
		vec_mk.push_back(vec_m[idx]*vec_k[idx]*vec_batchcnt[idx]);
		vec_nk.push_back(vec_n[idx]*vec_k[idx]*vec_batchcnt[idx]);
	}
	int maxPos_mn = std::max_element(vec_mn.begin(),vec_mn.end()) - vec_mn.begin();
	int maxPos_mk = std::max_element(vec_mk.begin(),vec_mk.end()) - vec_mk.begin();
	int maxPos_nk = std::max_element(vec_nk.begin(),vec_nk.end()) - vec_nk.begin();
	long long max_mn = vec_mn[maxPos_mn];
	long long max_mk = vec_mk[maxPos_mk];
	long long max_nk = vec_nk[maxPos_nk];

	long long memory_need = 0;

	memory_need = (max_mk + max_nk) * sizeof(Ti) + max_mn*2 * sizeof(To);
	cout << "Total lines: " << count << ", max_matrixA: " << sizeof(Ti)*max_mk / (1024.0*1024.0*1024.0) << ", max_matrixB: " << sizeof(Ti)*max_nk / (1024.0*1024.0*1024.0) \
		 << ", max_matrixC: " << sizeof(To)*max_mn / (1024.0*1024.0*1024.0) << ", max_matrixD: " << sizeof(To)*max_mn / (1024.0*1024.0*1024.0) \
		 << ", memory_need: " << memory_need / (1024.0*1024.0*1024.0)<< endl;	

	cout << "Check the current device memory status"<<endl;

	//check current device information
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, device);
        if (err != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        // Set the current device
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        err = hipMemGetInfo(&freeMem, &totalMem);
        if (err != hipSuccess) {
            std::cerr << "Failed to get memory info for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total memory: " << static_cast<double>(totalMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Free memory: " << static_cast<double>(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
 
	if (memory_need > freeMem) {
		std::cout << "Warning: Some test cases exceed the available memory on the device. Using the default memory allocation scheme."  << std::endl;
		const size_t G = 1024 * 1024 * 1024;
		float tmp = 0;
		size_t size_byte = sizeof(Ti) * 2 + sizeof(To) * 2;
		tmp = (std::floor((static_cast<double>(freeMem)/G/size_byte)*100))/100;

		max_mk = tmp * G;
		max_nk = tmp * G;
		max_mn = tmp * G;
		}
    cout << "Applying for memory:\n"
         << "Allocate memory for matrix A (GB): " << sizeof(Ti)*max_mk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix B (GB): " << sizeof(Ti)*max_nk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix C (GB): " << sizeof(To)*max_mn / 1024.0 / 1024.0 / 1024.0
		 << "\nAllocate memory for matrix D (GB): " << sizeof(To)*max_mn / 1024.0 / 1024.0 / 1024.0
         << endl;

    // Allocate host memory
	Ti *A, *B, *d_A, *d_B;
	To *C, *D, *d_C, *d_D;;
	 
	A = (Ti*)malloc(max_mk * sizeof(Ti));
	B = (Ti*)malloc(max_nk * sizeof(Ti));
	C = (To*)malloc(max_mn * sizeof(To));
	D = (To*)malloc(max_mn * sizeof(To));
	if (!A || !B || !C || !D) {
	cerr << "Failed to allocate host memory!" << endl;
	return EXIT_FAILURE;
	}
	// Allocate device memory
	CHECK_HIP_ERROR(hipMalloc((void**)&d_A, max_mk * sizeof(Ti)));
	CHECK_HIP_ERROR(hipMalloc((void**)&d_B, max_nk * sizeof(Ti)));
	CHECK_HIP_ERROR(hipMalloc((void**)&d_C, max_mn * sizeof(To)));
	CHECK_HIP_ERROR(hipMalloc((void**)&d_D, max_mn * sizeof(To)));
	
	// Initialize matrices

	#pragma omp parallel for
	for (size_t j = 0; j < max_mk; j++) {
		A[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
	}
	#pragma omp parallel for
	for (size_t j = 0; j < max_nk; j++) {
		B[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
	}
	#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
		C[j] = 0;
	}
	#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
		D[j] = 0;
	}

	std::cout << "Matrix initialization completed using OpenMP parallelization" << std::endl;

	//copy matrix to gpu
	CHECK_HIP_ERROR(hipMemcpy(d_A, A, max_mk * sizeof(Ti), hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemcpy(d_B, B, max_nk * sizeof(Ti), hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemcpy(d_C, C, max_mn * sizeof(To), hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemcpy(d_D, D, max_mn * sizeof(To), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    cout << "Copy matrix to GPU completed" << endl;
    float alpha_float = 1;
    float beta_float = 0;
    _Float16 alpha_half[2] = {1, 1};
    _Float16 beta_half[2] = {0, 0};
	
    void* alpha_ptr = nullptr;
    void* beta_ptr = nullptr;

	rocblas_datatype i, o, c;
	if (gemmType == "hpa")
	{	
		alpha_ptr = &alpha_float;
        beta_ptr = &beta_float;
		i = rocblas_datatype_f16_r;
		o = rocblas_datatype_f16_r;
		c = rocblas_datatype_f32_r;
	}
	else if (gemmType == "bf16")
	{
		alpha_ptr = &alpha_float;
        beta_ptr = &beta_float;
		i = rocblas_datatype_bf16_r;
		o = rocblas_datatype_bf16_r;
		c = rocblas_datatype_f32_r;
	}
	else if (gemmType == "hgemm")
	{
		alpha_ptr = alpha_half;  // 使用 _Float16 类型的 alpha
        beta_ptr = beta_half;
		i = rocblas_datatype_f16_r;
		o = rocblas_datatype_f16_r;
		c = rocblas_datatype_f16_r;
	}
	else if (gemmType == "int8")
	{
		alpha_ptr = &alpha_float;
        beta_ptr = &beta_float;
		i = rocblas_datatype_i8_r;
		o = rocblas_datatype_i32_r;
		c = rocblas_datatype_i32_r;
	}



	rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

	int updateInterval = (num_count + 99) / 100; // Update progress every 1% of completion
	for(int idx = 0; idx < num_count; idx++)
	{
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;

		m = vec_m[idx];
		n = vec_n[idx];
		k = vec_k[idx];
		batchCount = vec_batchcnt[idx];
		trans_a = vec_trans_a[idx];
		trans_b = vec_trans_b[idx];

		assert(trans_a == 'N' || trans_a == 'T' || trans_a == 'C');
		assert(trans_b == 'N' || trans_b == 'T' || trans_b == 'C');		
		size_t lda, ldb, ldc, ldd;
		if(trans_a == 'N'){
			lda = m;
		}
		else{
			lda = k;
		}
		if(trans_b == 'N'){
			ldb = k;
		}
		else{
			ldb = n;
		}
		ldc = m;
	    ldd =ldc;
        size_t stride_a = m * k;
        size_t stride_b = n * k;
        size_t stride_c = m * n;
        size_t stride_d = stride_c;

		hipStream_t stream;
		rocblas_get_stream(handle, &stream);

		if (m * k * batchCount > max_mk || n * k * batchCount > max_nk || m * n * batchCount > max_mn)
		{
			vec_gflops_results.push_back(-1);
			vec_time_costs_results.push_back(-1); // us
			failed_size_count++;
			continue;
		}
#ifdef WARMUP
		if (batchCount > 1){
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_gemm_strided_batched_ex(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					alpha_ptr,
					d_A, i,lda,stride_a,
					d_B, i,ldb,stride_b,
					beta_ptr,
					d_C, o,ldc,stride_c,
					d_D, o,ldd,stride_d,
					batchCount,
					c,rocblas_gemm_algo_standard,0,0);
			}
		}
		else {
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_gemm_ex(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					alpha_ptr,
					d_A, i,lda,
					d_B, i,ldb,
					beta_ptr,
					d_C, o,ldc,
					d_D, o,ldd,
					c,rocblas_gemm_algo_standard,0,0);
			}
		}
		// std::cout << std::endl << std::endl;
		// std::cout<<"warmup finished !"<< std::endl << std::endl;
		
#endif
		hipDeviceSynchronize();
		double gpu_time_used = get_time_us_sync(stream); // in microseconds

		if (batchCount > 1){
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_gemm_strided_batched_ex(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					alpha_ptr,
					d_A, i,lda,stride_a,
					d_B, i,ldb,stride_b,
					beta_ptr,
					d_C, o,ldc,stride_c,
					d_D, o,ldd,stride_d,
					batchCount,
					c,rocblas_gemm_algo_standard,0,0);
			}
		}
		else {
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_gemm_ex(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					alpha_ptr,
					d_A, i,lda,
					d_B, i,ldb,
					beta_ptr,
					d_C, o,ldc,
					d_D, o,ldd,
					c,rocblas_gemm_algo_standard,0,0);
			}
		}

		gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds
		double time_stage1 = gpu_time_used / 1000000.0;		
		double gemm_perf = 2.0 * 1e-9 * m * n * k * batchCount / (time_stage1 / iter_num);  // GFLOPS		

		vec_gflops_results.push_back(gemm_perf);
		vec_time_costs_results.push_back(gpu_time_used / iter_num); // us	
		if (idx % updateInterval == 0 || idx == num_count-1) { 
				updateProgressBar(idx, num_count);
       		 }
	}
    cout << endl << endl;


	// output files
	// write to csv file
	ofstream outFile;
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.open("Gemm_Generality_prof_origin.csv", ios::out);
	outFile << "trans_a" << ',' << "trans_b" << ',' << "M" << ',' << "N" << ','<< "B"<<',' << "K" << ',' <<"gflops"<<','<< "us" << endl;
	for(int idx = 0; idx < num_count; idx++)
	{
		std::cout << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ','<<vec_batchcnt[idx]<<',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << " gflops" << ',' << setprecision(6) << vec_time_costs_results[idx] << " us"<< endl;
		outFile << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ',' << vec_batchcnt[idx] << ',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << ',' << setprecision(6) << vec_time_costs_results[idx]<< endl;
	}
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.close();

    free(A);
    free(B);
    free(C);
    // free(D);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    // CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
	return 0;
}

int gemm_init_sgemm(char ** argv)
{
	ifstream inFile(argv[1]);
    if (!inFile.is_open()) {
		cerr << "Can't open the input file!" << endl;
        return EXIT_FAILURE;
	}

	int warmup_num = atoi(argv[2]);
	int iter_num = atoi(argv[3]);

	size_t max_m = 0;
	size_t max_n = 0;
	size_t max_k = 0;
	size_t max_batchCount = 0;
	size_t count = 0;

	std::vector<size_t> vec_m;
	std::vector<size_t> vec_n;
	std::vector<size_t> vec_k;
	std::vector<size_t> vec_batchcnt;
	std::vector<char> vec_trans_a;
	std::vector<char> vec_trans_b;

	std::vector<double> vec_gflops_results;
	std::vector<double> vec_time_costs_results;

	string line;
	while(getline(inFile, line))
	{
		// read input parameter
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;
		stringstream ss(line);
		ss >> trans_a;
		ss >> trans_b;
		ss >> m;
		ss >> n;
		ss >> k;
		ss >> batchCount;
		vec_trans_a.push_back(trans_a);
		vec_trans_b.push_back(trans_b);	
		vec_m.push_back(m);
		vec_n.push_back(n);
		vec_k.push_back(k);
		vec_batchcnt.push_back(batchCount);
																									 			
        max_m = max(max_m, m);
        max_n = max(max_n, n);
        max_k = max(max_k, k);
        max_batchCount = max(max_batchCount, batchCount);

		count++;
	}
    cout << "Total lines: " << count << ", max_m: " << max_m << ", max_n: " << max_n << ", max_k: " << max_k << ", max_batchCount: " << max_batchCount << endl;
	int num_count = count;
	inFile.close();


	std::vector<long long> vec_mn;
	std::vector<long long> vec_mk;
	std::vector<long long> vec_nk;	
	for(int idx = 0; idx < num_count; idx++)
	{
		vec_mn.push_back(vec_m[idx]*vec_n[idx]*vec_batchcnt[idx]);
		vec_mk.push_back(vec_m[idx]*vec_k[idx]*vec_batchcnt[idx]);
		vec_nk.push_back(vec_n[idx]*vec_k[idx]*vec_batchcnt[idx]);
	}
	int maxPos_mn = std::max_element(vec_mn.begin(),vec_mn.end()) - vec_mn.begin();
	int maxPos_mk = std::max_element(vec_mk.begin(),vec_mk.end()) - vec_mk.begin();
	int maxPos_nk = std::max_element(vec_nk.begin(),vec_nk.end()) - vec_nk.begin();
	long long max_mn = vec_mn[maxPos_mn];
	long long max_mk = vec_mk[maxPos_mk];
	long long max_nk = vec_nk[maxPos_nk];
	long long memory_need = (max_mk + max_nk + max_mn) * sizeof(float);
    cout << "Total lines: " << count << ", max_matrixA: " << sizeof(float)*max_mk / (1024.0*1024.0*1024.0) << ", max_matrixB: " << sizeof(float)*max_nk / (1024.0*1024.0*1024.0) << ", max_matrixC: " << sizeof(float)*max_mn / (1024.0*1024.0*1024.0) << ", memory_need: " << memory_need / (1024.0*1024.0*1024.0)<< endl;	
	cout << "Check the current device memory status"<<endl;

	//check current device information
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, device);
        if (err != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        // Set the current device
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        err = hipMemGetInfo(&freeMem, &totalMem);
        if (err != hipSuccess) {
            std::cerr << "Failed to get memory info for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total memory: " << static_cast<double>(totalMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Free memory: " << static_cast<double>(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
 
	if (memory_need > freeMem) {
		std::cout << "Warning: Some test cases exceed the available memory on the device. Using the default memory allocation scheme."  << std::endl;
		const size_t G = 1024 * 1024 * 1024;
		float tmp = (std::floor((static_cast<double>(freeMem)/G/3/sizeof(float))*100))/100;
		max_mk = tmp * G;
		max_nk = tmp * G;
		max_mn = tmp * G;
		}
    cout << "Applying for memory:\n"
         << "Allocate memory for matrix A (GB): " << sizeof(float)*max_mk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix B (GB): " << sizeof(float)*max_nk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix C (GB): " << sizeof(float)*max_mn / 1024.0 / 1024.0 / 1024.0
         << endl;

    // Allocate host memory
	// float *A, *B, *C, *D;
	float *A, *B, *C;
	A = (float*)malloc(max_mk * sizeof(float));
	B = (float*)malloc(max_nk * sizeof(float));
	C = (float*)malloc(max_mn * sizeof(float));
	// D = (float*)malloc(max_mn * sizeof(float));
    // if (!A || !B || !C || !D) {
		if (!A || !B || !C) {
        cerr << "Failed to allocate host memory!" << endl;
        return EXIT_FAILURE;
    }
	// Allocate device memory
	// float *d_A, *d_B, *d_C, *d_D;
	float *d_A, *d_B, *d_C;
	CHECK_HIP_ERROR(hipMalloc((void**)&d_A, max_mk * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_B, max_nk * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_C, max_mn * sizeof(float)));
    // CHECK_HIP_ERROR(hipMalloc((void**)&d_D, max_mn * sizeof(float)));
	
	// Initialize matrices

 	#pragma omp parallel for
    for (size_t j = 0; j < max_mk; j++) {
        A[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_nk; j++) {
        B[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_mn; j++) {
		C[j] = 0;
    }
	/*#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
    	D[j] = 0;
	}*/

	std::cout << "Matrix initialization completed using OpenMP parallelization" << std::endl;

	//copy matrix to gpu
	CHECK_HIP_ERROR(hipMemcpy(d_A, A, max_mk * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B, max_nk * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, C, max_mn * sizeof(float), hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(d_D, D, max_mn * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    cout << "Copy matrix to GPU completed" << endl;
	float alpha = 1;
    float beta = 0;

	rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

	int updateInterval = (num_count + 99) / 100; // Update progress every 1% of completion
	for(int idx = 0; idx < num_count; idx++)
	{
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;

		m = vec_m[idx];
		n = vec_n[idx];
		k = vec_k[idx];
		batchCount = vec_batchcnt[idx];
		trans_a = vec_trans_a[idx];
		trans_b = vec_trans_b[idx];

		assert(trans_a == 'N' || trans_a == 'T' || trans_a == 'n' || trans_a == 't');
		assert(trans_b == 'N' || trans_b == 'T' || trans_b == 'n' || trans_b == 't');
		size_t lda, ldb, ldc, ldd;
		if(trans_a == 'N'){
			lda = m;
		}
		else{
			lda = k;
		}
		if(trans_b == 'N'){
			ldb = k;
		}
		else{
			ldb = n;
		}
		ldc = m;
	    ldd =ldc;
        size_t stride_a = m * k;
        size_t stride_b = n * k;
        size_t stride_c = m * n;
        size_t stride_d = stride_c;

		hipStream_t stream;
		rocblas_get_stream(handle, &stream);

		if (m * k * batchCount > max_mk || n * k * batchCount > max_nk || m * n * batchCount > max_mn)
		{
			vec_gflops_results.push_back(-1);
			vec_time_costs_results.push_back(-1); // us
			failed_size_count++;
			continue;
		}


#ifdef WARMUP
		if (batchCount > 1){
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_sgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);
			}
		}
		else {
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_sgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
	// std::cout << std::endl << std::endl;
	// std::cout<<"warmup finished !"<< std::endl << std::endl;
	hipDeviceSynchronize();
#endif
        double gpu_time_used = get_time_us_sync(stream); // in microseconds

		if (batchCount > 1){
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_sgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);

			}
		}
		else {
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_sgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
		
		gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds
		double time_stage1 = gpu_time_used / 1000000.0;		
		double gemm_perf = 2.0 * 1e-9 * m * n * k * batchCount / (time_stage1 / iter_num);  // GFLOPS		

		vec_gflops_results.push_back(gemm_perf);
		vec_time_costs_results.push_back(gpu_time_used / iter_num); // us	
		if (idx % updateInterval == 0 || idx == num_count-1) { 
				updateProgressBar(idx, num_count);
       		 }
	}
    cout << endl << endl;


	// output files
	// write to csv file
	ofstream outFile;
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.open("Gemm_Generality_prof_origin.csv", ios::out);
	outFile << "trans_a" << ',' << "trans_b" << ',' << "M" << ',' << "N" << ','<< "B"<<',' << "K" << ',' <<"gflops"<<','<< "us" << endl;
	for(int idx = 0; idx < num_count; idx++)
	{
		std::cout << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ','<<vec_batchcnt[idx]<<',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << " gflops" << ',' << setprecision(6) << vec_time_costs_results[idx] << " us"<< endl;
		outFile << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ',' << vec_batchcnt[idx] << ',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << ',' << setprecision(6) << vec_time_costs_results[idx]<< endl;
	}
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.close();

    free(A);
    free(B);
    free(C);
    // free(D);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    // CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return 0;
}

int gemm_init_cgemm(char ** argv)
{
    ifstream inFile(argv[1]);
    if (!inFile.is_open()) {
		cerr << "Can't open the input file!" << endl;
        return EXIT_FAILURE;
	}

	int warmup_num = atoi(argv[2]);
	int iter_num = atoi(argv[3]);

	size_t max_m = 0;
	size_t max_n = 0;
	size_t max_k = 0;
	size_t max_batchCount = 0;
	size_t count = 0;

	std::vector<size_t> vec_m;
	std::vector<size_t> vec_n;
	std::vector<size_t> vec_k;
	std::vector<size_t> vec_batchcnt;
	std::vector<char> vec_trans_a;
	std::vector<char> vec_trans_b;

	std::vector<double> vec_gflops_results;
	std::vector<double> vec_time_costs_results;

	string line;
	while(getline(inFile, line))
	{
		// read input parameter
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;
		stringstream ss(line);
		ss >> trans_a;
		ss >> trans_b;
		ss >> m;
		ss >> n;
		ss >> k;
		ss >> batchCount;
		vec_trans_a.push_back(trans_a);
		vec_trans_b.push_back(trans_b);	
		vec_m.push_back(m);
		vec_n.push_back(n);
		vec_k.push_back(k);
		vec_batchcnt.push_back(batchCount);
																									 			
        max_m = max(max_m, m);
        max_n = max(max_n, n);
        max_k = max(max_k, k);
        max_batchCount = max(max_batchCount, batchCount);

		count++;
	}
    cout << "Total lines: " << count << ", max_m: " << max_m << ", max_n: " << max_n << ", max_k: " << max_k << ", max_batchCount: " << max_batchCount << endl;
	int num_count = count;
	inFile.close();


	std::vector<long long> vec_mn;
	std::vector<long long> vec_mk;
	std::vector<long long> vec_nk;	
	for(int idx = 0; idx < num_count; idx++)
	{
		vec_mn.push_back(vec_m[idx]*vec_n[idx]*vec_batchcnt[idx]);
		vec_mk.push_back(vec_m[idx]*vec_k[idx]*vec_batchcnt[idx]);
		vec_nk.push_back(vec_n[idx]*vec_k[idx]*vec_batchcnt[idx]);
	}
	int maxPos_mn = std::max_element(vec_mn.begin(),vec_mn.end()) - vec_mn.begin();
	int maxPos_mk = std::max_element(vec_mk.begin(),vec_mk.end()) - vec_mk.begin();
	int maxPos_nk = std::max_element(vec_nk.begin(),vec_nk.end()) - vec_nk.begin();
	long long max_mn = vec_mn[maxPos_mn];
	long long max_mk = vec_mk[maxPos_mk];
	long long max_nk = vec_nk[maxPos_nk];
	long long memory_need = (max_mk + max_nk + max_mn) * sizeof(rocblas_float_complex);
    cout << "Total lines: " << count << ", max_matrixA: " << sizeof(rocblas_float_complex)*max_mk / (1024.0*1024.0*1024.0) << ", max_matrixB: " << sizeof(rocblas_float_complex)*max_nk / (1024.0*1024.0*1024.0) << ", max_matrixC: " << sizeof(rocblas_float_complex)*max_mn / (1024.0*1024.0*1024.0) << ", memory_need: " << memory_need / (1024.0*1024.0*1024.0)<< endl;	
	cout << "Check the current device memory status"<<endl;

	//check current device information
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, device);
        if (err != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        // Set the current device
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        err = hipMemGetInfo(&freeMem, &totalMem);
        if (err != hipSuccess) {
            std::cerr << "Failed to get memory info for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total memory: " << static_cast<double>(totalMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Free memory: " << static_cast<double>(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
 
	if (memory_need > freeMem) {
		std::cout << "Warning: Some test cases exceed the available memory on the device. Using the default memory allocation scheme."  << std::endl;
		const size_t G = 1024 * 1024 * 1024;
		float tmp = (std::floor((static_cast<double>(freeMem)/G/3/sizeof(rocblas_float_complex))*100))/100;
		max_mk = tmp * G;
		max_nk = tmp * G;
		max_mn = tmp * G;
		}
    cout << "Applying for memory:\n"
         << "Allocate memory for matrix A (GB): " << sizeof(rocblas_float_complex)*max_mk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix B (GB): " << sizeof(rocblas_float_complex)*max_nk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix C (GB): " << sizeof(rocblas_float_complex)*max_mn / 1024.0 / 1024.0 / 1024.0
         << endl;

    // Allocate host memory
	// rocblas_float_complex *A, *B, *C, *D;
	rocblas_float_complex *A, *B, *C;
	A = (rocblas_float_complex*)malloc(max_mk * sizeof(rocblas_float_complex));
	B = (rocblas_float_complex*)malloc(max_nk * sizeof(rocblas_float_complex));
	C = (rocblas_float_complex*)malloc(max_mn * sizeof(rocblas_float_complex));
	// D = (rocblas_float_complex*)malloc(max_mn * sizeof(rocblas_float_complex));
    // if (!A || !B || !C || !D) {
		if (!A || !B || !C) {
        cerr << "Failed to allocate host memory!" << endl;
        return EXIT_FAILURE;
    }
	// Allocate device memory
	// rocblas_float_complex *d_A, *d_B, *d_C, *d_D;
	rocblas_float_complex *d_A, *d_B, *d_C;
	CHECK_HIP_ERROR(hipMalloc((void**)&d_A, max_mk * sizeof(rocblas_float_complex)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_B, max_nk * sizeof(rocblas_float_complex)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_C, max_mn * sizeof(rocblas_float_complex)));
    // CHECK_HIP_ERROR(hipMalloc((void**)&d_D, max_mn * sizeof(rocblas_float_complex)));
	
	// Initialize matrices

 	#pragma omp parallel for
    for (size_t j = 0; j < max_mk; j++) {
        ((A[j]).x) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		((A[j]).y) = 0.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_nk; j++) {
        ((B[j]).x) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		((B[j]).y) = 0.0f; 
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_mn; j++) {
		((C[j]).x) = 0.0f;
		((C[j]).y) = 0.0f;
    }
	/*#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
    	D[j].x = 0.0f;
		D[j].y = 0.0f;
	}*/

	std::cout << "Matrix initialization completed using OpenMP parallelization" << std::endl;

	//copy matrix to gpu
	CHECK_HIP_ERROR(hipMemcpy(d_A, A, max_mk * sizeof(rocblas_float_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B, max_nk * sizeof(rocblas_float_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, C, max_mn * sizeof(rocblas_float_complex), hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(d_D, D, max_mn * sizeof(rocblas_float_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    cout << "Copy matrix to GPU completed" << endl;
    rocblas_float_complex alpha;
	alpha.x = 1.0;
	alpha.y = 0;
    rocblas_float_complex beta;
	beta.x  = 0.0;
	beta.y  = 0.0;

	rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

	int updateInterval = (num_count + 99) / 100; // Update progress every 1% of completion
	for(int idx = 0; idx < num_count; idx++)
	{
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;

		m = vec_m[idx];
		n = vec_n[idx];
		k = vec_k[idx];
		batchCount = vec_batchcnt[idx];
		trans_a = vec_trans_a[idx];
		trans_b = vec_trans_b[idx];

		assert(trans_a == 'N' || trans_a == 'T' || trans_a == 'C');
		assert(trans_b == 'N' || trans_b == 'T' || trans_b == 'C');		
		size_t lda, ldb, ldc, ldd;
		if(trans_a == 'N'){
			lda = m;
		}
		else{
			lda = k;
		}
		if(trans_b == 'N'){
			ldb = k;
		}
		else{
			ldb = n;
		}
		ldc = m;
	    ldd =ldc;
        size_t stride_a = m * k;
        size_t stride_b = n * k;
        size_t stride_c = m * n;
        size_t stride_d = stride_c;

		hipStream_t stream;
		rocblas_get_stream(handle, &stream);

		if (m * k * batchCount > max_mk || n * k * batchCount > max_nk || m * n * batchCount > max_mn)
		{
			vec_gflops_results.push_back(-1);
			vec_time_costs_results.push_back(-1); // us
			failed_size_count++;
			continue;
		}


#ifdef WARMUP
		if (batchCount > 1){
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_cgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);
			}
		}
		else {
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_cgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
	// std::cout << std::endl << std::endl;
	// std::cout<<"warmup finished !"<< std::endl << std::endl;
	hipDeviceSynchronize();
#endif
        double gpu_time_used = get_time_us_sync(stream); // in microseconds

		if (batchCount > 1){
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_cgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);

			}
		}
		else {
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_cgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
		
		gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds
		double time_stage1 = gpu_time_used / 1000000.0;		
		double gemm_perf = 8.0 * 1e-9 * m * n * k * batchCount / (time_stage1 / iter_num);  // GFLOPS		

		vec_gflops_results.push_back(gemm_perf);
		vec_time_costs_results.push_back(gpu_time_used / iter_num); // us	
		if (idx % updateInterval == 0 || idx == num_count-1) { 
				updateProgressBar(idx, num_count);
       		 }
	}
    cout << endl << endl;


	// output files
	// write to csv file
	ofstream outFile;
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.open("Gemm_Generality_prof_origin.csv", ios::out);
	outFile << "trans_a" << ',' << "trans_b" << ',' << "M" << ',' << "N" << ','<< "B"<<',' << "K" << ',' <<"gflops"<<','<< "us" << endl;
	for(int idx = 0; idx < num_count; idx++)
	{
		std::cout << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ','<<vec_batchcnt[idx]<<',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << " gflops" << ',' << setprecision(6) << vec_time_costs_results[idx] << " us"<< endl;
		outFile << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ',' << vec_batchcnt[idx] << ',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << ',' << setprecision(6) << vec_time_costs_results[idx]<< endl;
	}
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.close();

    free(A);
    free(B);
    free(C);
    // free(D);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    // CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return 0;	
}

int gemm_init_dgemm(char ** argv)
{
    ifstream inFile(argv[1]);
    if (!inFile.is_open()) {
		cerr << "Can't open the input file!" << endl;
        return EXIT_FAILURE;
	}

	int warmup_num = atoi(argv[2]);
	int iter_num = atoi(argv[3]);

	size_t max_m = 0;
	size_t max_n = 0;
	size_t max_k = 0;
	size_t max_batchCount = 0;
	size_t count = 0;

	std::vector<size_t> vec_m;
	std::vector<size_t> vec_n;
	std::vector<size_t> vec_k;
	std::vector<size_t> vec_batchcnt;
	std::vector<char> vec_trans_a;
	std::vector<char> vec_trans_b;

	std::vector<double> vec_gflops_results;
	std::vector<double> vec_time_costs_results;

	string line;
	while(getline(inFile, line))
	{
		// read input parameter
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;
		stringstream ss(line);
		ss >> trans_a;
		ss >> trans_b;
		ss >> m;
		ss >> n;
		ss >> k;
		ss >> batchCount;
		vec_trans_a.push_back(trans_a);
		vec_trans_b.push_back(trans_b);	
		vec_m.push_back(m);
		vec_n.push_back(n);
		vec_k.push_back(k);
		vec_batchcnt.push_back(batchCount);
																									 			
        max_m = max(max_m, m);
        max_n = max(max_n, n);
        max_k = max(max_k, k);
        max_batchCount = max(max_batchCount, batchCount);

		count++;
	}
    cout << "Total lines: " << count << ", max_m: " << max_m << ", max_n: " << max_n << ", max_k: " << max_k << ", max_batchCount: " << max_batchCount << endl;
	int num_count = count;
	inFile.close();


	std::vector<long long> vec_mn;
	std::vector<long long> vec_mk;
	std::vector<long long> vec_nk;	
	for(int idx = 0; idx < num_count; idx++)
	{
		vec_mn.push_back(vec_m[idx]*vec_n[idx]*vec_batchcnt[idx]);
		vec_mk.push_back(vec_m[idx]*vec_k[idx]*vec_batchcnt[idx]);
		vec_nk.push_back(vec_n[idx]*vec_k[idx]*vec_batchcnt[idx]);
	}
	int maxPos_mn = std::max_element(vec_mn.begin(),vec_mn.end()) - vec_mn.begin();
	int maxPos_mk = std::max_element(vec_mk.begin(),vec_mk.end()) - vec_mk.begin();
	int maxPos_nk = std::max_element(vec_nk.begin(),vec_nk.end()) - vec_nk.begin();
	long long max_mn = vec_mn[maxPos_mn];
	long long max_mk = vec_mk[maxPos_mk];
	long long max_nk = vec_nk[maxPos_nk];
	long long memory_need = (max_mk + max_nk + max_mn) * sizeof(double);
    cout << "Total lines: " << count << ", max_matrixA: " << sizeof(double)*max_mk / (1024.0*1024.0*1024.0) << ", max_matrixB: " << sizeof(double)*max_nk / (1024.0*1024.0*1024.0) << ", max_matrixC: " << sizeof(double)*max_mn / (1024.0*1024.0*1024.0) << ", memory_need: " << memory_need / (1024.0*1024.0*1024.0)<< endl;	
	cout << "Check the current device memory status"<<endl;

	//check current device information
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, device);
        if (err != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        // Set the current device
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        err = hipMemGetInfo(&freeMem, &totalMem);
        if (err != hipSuccess) {
            std::cerr << "Failed to get memory info for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total memory: " << static_cast<double>(totalMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Free memory: " << static_cast<double>(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
 
	if (memory_need > freeMem) {
		std::cout << "Warning: Some test cases exceed the available memory on the device. Using the default memory allocation scheme."  << std::endl;
		const size_t G = 1024 * 1024 * 1024;
		float tmp = (std::floor((static_cast<double>(freeMem)/G/3/sizeof(double))*100))/100;
		max_mk = tmp * G;
		max_nk = tmp * G;
		max_mn = tmp * G;
		}
    cout << "Applying for memory:\n"
         << "Allocate memory for matrix A (GB): " << sizeof(double)*max_mk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix B (GB): " << sizeof(double)*max_nk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix C (GB): " << sizeof(double)*max_mn / 1024.0 / 1024.0 / 1024.0
         << endl;

    // Allocate host memory
	// double *A, *B, *C, *D;
	double *A, *B, *C;
	A = (double*)malloc(max_mk * sizeof(double));
	B = (double*)malloc(max_nk * sizeof(double));
	C = (double*)malloc(max_mn * sizeof(double));
	// D = (double*)malloc(max_mn * sizeof(double));
    // if (!A || !B || !C || !D) {
		if (!A || !B || !C) {
        cerr << "Failed to allocate host memory!" << endl;
        return EXIT_FAILURE;
    }
	// Allocate device memory
	// double *d_A, *d_B, *d_C, *d_D;
	double *d_A, *d_B, *d_C;
	CHECK_HIP_ERROR(hipMalloc((void**)&d_A, max_mk * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_B, max_nk * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_C, max_mn * sizeof(double)));
    // CHECK_HIP_ERROR(hipMalloc((void**)&d_D, max_mn * sizeof(double)));
	
	// Initialize matrices

 	#pragma omp parallel for
    for (size_t j = 0; j < max_mk; j++) {
        A[j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 2.0f - 1.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_nk; j++) {
        B[j] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 2.0f - 1.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_mn; j++) {
		C[j] = 0;
    }
	/*#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
    	D[j] = 0;
	}*/

	std::cout << "Matrix initialization completed using OpenMP parallelization" << std::endl;

	//copy matrix to gpu
	CHECK_HIP_ERROR(hipMemcpy(d_A, A, max_mk * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B, max_nk * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, C, max_mn * sizeof(double), hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(d_D, D, max_mn * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    cout << "Copy matrix to GPU completed" << endl;
	double alpha = 1;
    double beta = 0;

	rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

	int updateInterval = (num_count + 99) / 100; // Update progress every 1% of completion
	for(int idx = 0; idx < num_count; idx++)
	{
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;

		m = vec_m[idx];
		n = vec_n[idx];
		k = vec_k[idx];
		batchCount = vec_batchcnt[idx];
		trans_a = vec_trans_a[idx];
		trans_b = vec_trans_b[idx];

		assert(trans_a == 'N' || trans_a == 'T' || trans_a == 'n' || trans_a == 't');
		assert(trans_b == 'N' || trans_b == 'T' || trans_b == 'n' || trans_b == 't');
		size_t lda, ldb, ldc, ldd;
		if(trans_a == 'N'){
			lda = m;
		}
		else{
			lda = k;
		}
		if(trans_b == 'N'){
			ldb = k;
		}
		else{
			ldb = n;
		}
		ldc = m;
	    ldd =ldc;
        size_t stride_a = m * k;
        size_t stride_b = n * k;
        size_t stride_c = m * n;
        size_t stride_d = stride_c;

		hipStream_t stream;
		rocblas_get_stream(handle, &stream);

		if (m * k * batchCount > max_mk || n * k * batchCount > max_nk || m * n * batchCount > max_mn)
		{
			vec_gflops_results.push_back(-1);
			vec_time_costs_results.push_back(-1); // us
			failed_size_count++;
			continue;
		}


#ifdef WARMUP
		if (batchCount > 1){
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_dgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);
			}
		}
		else {
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_dgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
	// std::cout << std::endl << std::endl;
	// std::cout<<"warmup finished !"<< std::endl << std::endl;
	hipDeviceSynchronize();
#endif
        double gpu_time_used = get_time_us_sync(stream); // in microseconds

		if (batchCount > 1){
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_dgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);

			}
		}
		else {
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_dgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
		
		gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds
		double time_stage1 = gpu_time_used / 1000000.0;		
		double gemm_perf = 2.0 * 1e-9 * m * n * k * batchCount / (time_stage1 / iter_num);  // GFLOPS		

		vec_gflops_results.push_back(gemm_perf);
		vec_time_costs_results.push_back(gpu_time_used / iter_num); // us	
		if (idx % updateInterval == 0 || idx == num_count-1) { 
				updateProgressBar(idx, num_count);
       		 }
	}
    cout << endl << endl;


	// output files
	// write to csv file
	ofstream outFile;
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.open("Gemm_Generality_prof_origin.csv", ios::out);
	outFile << "trans_a" << ',' << "trans_b" << ',' << "M" << ',' << "N" << ','<< "B"<<',' << "K" << ',' <<"gflops"<<','<< "us" << endl;
	for(int idx = 0; idx < num_count; idx++)
	{
		std::cout << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ','<<vec_batchcnt[idx]<<',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << " gflops" << ',' << setprecision(6) << vec_time_costs_results[idx] << " us"<< endl;
		outFile << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ',' << vec_batchcnt[idx] << ',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << ',' << setprecision(6) << vec_time_costs_results[idx]<< endl;
	}
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.close();

    free(A);
    free(B);
    free(C);
    // free(D);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    // CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return 0;	
}

int gemm_init_zgemm(char ** argv)
{
    ifstream inFile(argv[1]);
    if (!inFile.is_open()) {
		cerr << "Can't open the input file!" << endl;
        return EXIT_FAILURE;
	}

	int warmup_num = atoi(argv[2]);
	int iter_num = atoi(argv[3]);

	size_t max_m = 0;
	size_t max_n = 0;
	size_t max_k = 0;
	size_t max_batchCount = 0;
	size_t count = 0;

	std::vector<size_t> vec_m;
	std::vector<size_t> vec_n;
	std::vector<size_t> vec_k;
	std::vector<size_t> vec_batchcnt;
	std::vector<char> vec_trans_a;
	std::vector<char> vec_trans_b;

	std::vector<double> vec_gflops_results;
	std::vector<double> vec_time_costs_results;

	string line;
	while(getline(inFile, line))
	{
		// read input parameter
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;
		stringstream ss(line);
		ss >> trans_a;
		ss >> trans_b;
		ss >> m;
		ss >> n;
		ss >> k;
		ss >> batchCount;
		vec_trans_a.push_back(trans_a);
		vec_trans_b.push_back(trans_b);	
		vec_m.push_back(m);
		vec_n.push_back(n);
		vec_k.push_back(k);
		vec_batchcnt.push_back(batchCount);
																									 			
        max_m = max(max_m, m);
        max_n = max(max_n, n);
        max_k = max(max_k, k);
        max_batchCount = max(max_batchCount, batchCount);

		count++;
	}
    cout << "Total lines: " << count << ", max_m: " << max_m << ", max_n: " << max_n << ", max_k: " << max_k << ", max_batchCount: " << max_batchCount << endl;
	int num_count = count;
	inFile.close();


	std::vector<long long> vec_mn;
	std::vector<long long> vec_mk;
	std::vector<long long> vec_nk;	
	for(int idx = 0; idx < num_count; idx++)
	{
		vec_mn.push_back(vec_m[idx]*vec_n[idx]*vec_batchcnt[idx]);
		vec_mk.push_back(vec_m[idx]*vec_k[idx]*vec_batchcnt[idx]);
		vec_nk.push_back(vec_n[idx]*vec_k[idx]*vec_batchcnt[idx]);
	}
	int maxPos_mn = std::max_element(vec_mn.begin(),vec_mn.end()) - vec_mn.begin();
	int maxPos_mk = std::max_element(vec_mk.begin(),vec_mk.end()) - vec_mk.begin();
	int maxPos_nk = std::max_element(vec_nk.begin(),vec_nk.end()) - vec_nk.begin();
	long long max_mn = vec_mn[maxPos_mn];
	long long max_mk = vec_mk[maxPos_mk];
	long long max_nk = vec_nk[maxPos_nk];
	long long memory_need = (max_mk + max_nk + max_mn) * sizeof(rocblas_double_complex);
    cout << "Total lines: " << count << ", max_matrixA: " << sizeof(rocblas_double_complex)*max_mk / (1024.0*1024.0*1024.0) << ", max_matrixB: " << sizeof(rocblas_double_complex)*max_nk / (1024.0*1024.0*1024.0) << ", max_matrixC: " << sizeof(rocblas_double_complex)*max_mn / (1024.0*1024.0*1024.0) << ", memory_need: " << memory_need / (1024.0*1024.0*1024.0)<< endl;	
	cout << "Check the current device memory status"<<endl;

	//check current device information
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, device);
        if (err != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        // Set the current device
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            std::cerr << "Failed to set device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        err = hipMemGetInfo(&freeMem, &totalMem);
        if (err != hipSuccess) {
            std::cerr << "Failed to get memory info for device " << device << ": " << hipGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total memory: " << static_cast<double>(totalMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Free memory: " << static_cast<double>(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
 
	if (memory_need > freeMem) {
		std::cout << "Warning: Some test cases exceed the available memory on the device. Using the default memory allocation scheme."  << std::endl;
		const size_t G = 1024 * 1024 * 1024;
		float tmp = (std::floor((static_cast<double>(freeMem)/G/3/sizeof(rocblas_double_complex))*100))/100;
		max_mk = tmp * G;
		max_nk = tmp * G;
		max_mn = tmp * G;
		}
    cout << "Applying for memory:\n"
         << "Allocate memory for matrix A (GB): " << sizeof(rocblas_double_complex)*max_mk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix B (GB): " << sizeof(rocblas_double_complex)*max_nk / 1024.0 / 1024.0 / 1024.0
         << "\nAllocate memory for matrix C (GB): " << sizeof(rocblas_double_complex)*max_mn / 1024.0 / 1024.0 / 1024.0
         << endl;

    // Allocate host memory
	// rocblas_double_complex *A, *B, *C, *D;
	rocblas_double_complex *A, *B, *C;
	A = (rocblas_double_complex*)malloc(max_mk * sizeof(rocblas_double_complex));
	B = (rocblas_double_complex*)malloc(max_nk * sizeof(rocblas_double_complex));
	C = (rocblas_double_complex*)malloc(max_mn * sizeof(rocblas_double_complex));
	// D = (rocblas_double_complex*)malloc(max_mn * sizeof(rocblas_double_complex));
    // if (!A || !B || !C || !D) {
		if (!A || !B || !C) {
        cerr << "Failed to allocate host memory!" << endl;
        return EXIT_FAILURE;
    }
	// Allocate device memory
	// rocblas_double_complex *d_A, *d_B, *d_C, *d_D;
	rocblas_double_complex *d_A, *d_B, *d_C;
	CHECK_HIP_ERROR(hipMalloc((void**)&d_A, max_mk * sizeof(rocblas_double_complex)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_B, max_nk * sizeof(rocblas_double_complex)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_C, max_mn * sizeof(rocblas_double_complex)));
    // CHECK_HIP_ERROR(hipMalloc((void**)&d_D, max_mn * sizeof(rocblas_double_complex)));
	
	// Initialize matrices

 	#pragma omp parallel for
    for (size_t j = 0; j < max_mk; j++) {
        ((A[j]).x) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		((A[j]).y) = 0.0f;
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_nk; j++) {
        ((B[j]).x) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		((B[j]).y) = 0.0f; 
    }
	#pragma omp parallel for
    for (size_t j = 0; j < max_mn; j++) {
		((C[j]).x) = 0.0f;
		((C[j]).y) = 0.0f;
    }
	/*#pragma omp parallel for
	for (size_t j = 0; j < max_mn; j++) {
    	D[j].x = 0.0f;
		D[j].y = 0.0f;
	}*/

	std::cout << "Matrix initialization completed using OpenMP parallelization" << std::endl;

	//copy matrix to gpu
	CHECK_HIP_ERROR(hipMemcpy(d_A, A, max_mk * sizeof(rocblas_double_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B, max_nk * sizeof(rocblas_double_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, C, max_mn * sizeof(rocblas_double_complex), hipMemcpyHostToDevice));
    // CHECK_HIP_ERROR(hipMemcpy(d_D, D, max_mn * sizeof(rocblas_double_complex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    cout << "Copy matrix to GPU completed" << endl;
    rocblas_double_complex alpha;
	alpha.x = 1.0;
	alpha.y = 0.0;
    rocblas_double_complex beta;
	beta.x  = 0.0;
	beta.y  = 0.0;

	rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

	int updateInterval = (num_count + 99) / 100; // Update progress every 1% of completion
	for(int idx = 0; idx < num_count; idx++)
	{
		size_t m;
		size_t n;
		size_t k;
		size_t batchCount;
		char trans_a;
		char trans_b;

		m = vec_m[idx];
		n = vec_n[idx];
		k = vec_k[idx];
		batchCount = vec_batchcnt[idx];
		trans_a = vec_trans_a[idx];
		trans_b = vec_trans_b[idx];

		assert(trans_a == 'N' || trans_a == 'T' || trans_a == 'C');
		assert(trans_b == 'N' || trans_b == 'T' || trans_b == 'C');		
		size_t lda, ldb, ldc, ldd;
		if(trans_a == 'N'){
			lda = m;
		}
		else{
			lda = k;
		}
		if(trans_b == 'N'){
			ldb = k;
		}
		else{
			ldb = n;
		}
		ldc = m;
	    ldd =ldc;
        size_t stride_a = m * k;
        size_t stride_b = n * k;
        size_t stride_c = m * n;
        size_t stride_d = stride_c;

		hipStream_t stream;
		rocblas_get_stream(handle, &stream);

		if (m * k * batchCount > max_mk || n * k * batchCount > max_nk || m * n * batchCount > max_mn)
		{
			vec_gflops_results.push_back(-1);
			vec_time_costs_results.push_back(-1); // us
			failed_size_count++;
			continue;
		}


#ifdef WARMUP
		if (batchCount > 1){
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_zgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);
			}
		}
		else {
			for(int index = 0; index < warmup_num; index++)
			{
				rocblas_zgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
	// std::cout << std::endl << std::endl;
	// std::cout<<"warmup finished !"<< std::endl << std::endl;
	hipDeviceSynchronize();
#endif
        double gpu_time_used = get_time_us_sync(stream); // in microseconds

		if (batchCount > 1){
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_zgemm_strided_batched(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,stride_a,
					d_B, ldb,stride_b,
					&beta,
					d_C, ldc,stride_c,
					batchCount);

			}
		}
		else {
			for(int index = 0; index < iter_num; index++)
			{
				rocblas_zgemm(handle,
					getTranspose(trans_a), getTranspose(trans_b),
					m, n, k,
					&alpha,
					d_A, lda,
					d_B, ldb,
					&beta,
					d_C, ldc);
			}
		}
		
		gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds
		double time_stage1 = gpu_time_used / 1000000.0;		
		double gemm_perf = 8.0 * 1e-9 * m * n * k * batchCount / (time_stage1 / iter_num);  // GFLOPS		

		vec_gflops_results.push_back(gemm_perf);
		vec_time_costs_results.push_back(gpu_time_used / iter_num); // us	
		if (idx % updateInterval == 0 || idx == num_count-1) { 
				updateProgressBar(idx, num_count);
       		 }
	}
    cout << endl << endl;


	// output files
	// write to csv file
	ofstream outFile;
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.open("Gemm_Generality_prof_origin.csv", ios::out);
	outFile << "trans_a" << ',' << "trans_b" << ',' << "M" << ',' << "N" << ','<< "B"<<',' << "K" << ',' <<"gflops"<<','<< "us" << endl;
	for(int idx = 0; idx < num_count; idx++)
	{
		std::cout << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ','<<vec_batchcnt[idx]<<',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << " gflops" << ',' << setprecision(6) << vec_time_costs_results[idx] << " us"<< endl;
		outFile << vec_trans_a[idx] << ',' << vec_trans_b[idx] << ',' << vec_m[idx] << ',' << vec_n[idx] << ',' << vec_batchcnt[idx] << ',' << vec_k[idx]  << ',' << setprecision(6) << vec_gflops_results[idx] << ',' << setprecision(6) << vec_time_costs_results[idx]<< endl;
	}
	if (failed_size_count !=0) {cout<<"Warning!!! The number of sizes that failed the test is : "<<failed_size_count<<endl;}
	outFile.close();

    free(A);
    free(B);
    free(C);
    // free(D);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    // CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return 0;	
}

int main(int argc, char* argv[]) {
    if (argc < 5) { 
		std::cerr << "Usage: " << argv[0] << " <input_file> <warmup_num> <iter_num> <gemm_./atype> " << std::endl;
        return EXIT_FAILURE;
    }


	string gemmType(argv[4]);
	if (gemmType == "sgemm") 
	{
		gemm_init_sgemm(argv);
	}
	else if (gemmType == "hpa") 
	// if (gemmType == "hpa") 
	{
		gemm_init_gemm<_Float16,_Float16,float>(argv,gemmType);
	} 
	else if (gemmType == "bf16") 
	{
		gemm_init_gemm<_Float16,_Float16,float>(argv,gemmType);
	} 
	else if (gemmType == "hgemm") 
	{
		gemm_init_gemm<_Float16,_Float16,_Float16>(argv,gemmType);
	} 
	else if (gemmType == "cgemm") 
	{
		gemm_init_cgemm(argv);
	} 
	else if (gemmType == "dgemm") 
	{
		gemm_init_dgemm(argv);
	} 
	else if (gemmType == "zgemm") 
	{
		gemm_init_zgemm(argv);
	} 
	else if (gemmType == "int8") 
	{
		gemm_init_gemm<int8_t,int32_t,int32_t>(argv,gemmType);
	} 
	else 
	{
		std::cerr << "Unknown GEMM type" << std::endl;
		exit(EXIT_FAILURE);
	}
	// gemm_init(argv)
    return 0;
}