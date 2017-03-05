#include "bilateral_grid_simplified.h"

bilateral_grid_simplified::bilateral_grid_simplified() :
	nb_vertices(0),
	nb_reference_pixels(0),
	reference_width(0),
	reference_height(0)
{

}

void bilateral_grid_simplified::init(const cv::Mat reference_bgr, const int sigma_spatial, const int sigma_luma, const int sigma_chroma)
{
	cv::Mat reference_yuv;
	cv::cvtColor(reference_bgr, reference_yuv, CV_BGR2YUV);

	std::chrono::steady_clock::time_point begin_grid_construction = std::chrono::steady_clock::now();

	const int w = reference_yuv.cols;
	const int h = reference_yuv.rows;

	reference_width = w;
	reference_height = h;
	nb_reference_pixels = w * h;

	int max_coord[5];
	max_coord[0] = w / sigma_spatial;
	max_coord[1] = h / sigma_spatial;
	max_coord[2] = 255 / sigma_luma;
	max_coord[3] = 255 / sigma_chroma;
	max_coord[4] = 255 / sigma_chroma;

	std::int64_t hash_vec[5];
	for (int i = 0; i < 5; ++i)
		hash_vec[i] = static_cast<std::int64_t>(std::pow(255, i));

	std::unordered_map<std::int64_t /* hash */, int /* vert id */> hashed_coords;
	hashed_coords.reserve(w*h);

	const unsigned char* pref = (const unsigned char*)reference_yuv.data;
	int vert_idx = 0;
	int pix_idx = 0;

	typedef Eigen::Triplet<float> T;
	std::vector<T> tripletList;
	tripletList.reserve(w * h);

	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			std::int64_t coord[5];
			coord[0] = x / sigma_spatial;
			coord[1] = y / sigma_spatial;
			coord[2] = pref[0] / sigma_luma;
			coord[3] = pref[1] / sigma_chroma;
			coord[4] = pref[2] / sigma_chroma;

			std::int64_t hash_coord = 0;
			for (int i = 0; i < 5; ++i)
				hash_coord += coord[i] * hash_vec[i];

			auto it = hashed_coords.find(hash_coord);
			if (it == hashed_coords.end())
			{
				hashed_coords.insert(std::pair<std::int64_t, int>(hash_coord, vert_idx));
				tripletList.push_back(T(vert_idx, pix_idx, 1.0f));
				++vert_idx;
			}
			else
			{
				tripletList.push_back(T(it->second, pix_idx, 1.0f));

			}

			pref += 3;
			++pix_idx;
		}
	}


	mat_splat = Eigen::SparseMatrix<float, Eigen::RowMajor>(hashed_coords.size(), tripletList.size());
	mat_splat.setFromTriplets(tripletList.begin(), tripletList.end());
	mat_slice = mat_splat.transpose();

	nb_vertices = static_cast<std::int32_t>(hashed_coords.size());
	std::chrono::steady_clock::time_point end_grid_construction = std::chrono::steady_clock::now();
	std::cout << "grid construction:" << std::chrono::duration_cast<std::chrono::milliseconds>(end_grid_construction - begin_grid_construction).count() << "ms" << std::endl;


	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();

	// Blur matrices
	Eigen::SparseMatrix<float, Eigen::RowMajor> mat_b_left(hashed_coords.size(), hashed_coords.size());
	Eigen::SparseMatrix<float, Eigen::RowMajor> mat_b_right(hashed_coords.size(), hashed_coords.size());
	mat_blur = Eigen::SparseMatrix<float, Eigen::RowMajor>(hashed_coords.size(), hashed_coords.size());
	for (int i = 0; i < 5; ++i)
	{
		std::int64_t offset_hash_coord = -1 * hash_vec[i];

		tripletList.clear();
		for (auto it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
		{
			std::int64_t neighb_coord = it->first + offset_hash_coord;
			auto it_neighb = hashed_coords.find(neighb_coord);
			if (it_neighb != hashed_coords.end())
			{
				tripletList.push_back(T(it->second, it_neighb->second, 1.0f));
			}

		}
		mat_b_left.setZero();
		mat_b_left.setFromTriplets(tripletList.begin(), tripletList.end());


		offset_hash_coord = 1 * hash_vec[i];

		tripletList.clear();
		for (auto it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
		{
			std::int64_t neighb_coord = it->first + offset_hash_coord;
			auto it_neighb = hashed_coords.find(neighb_coord);
			if (it_neighb != hashed_coords.end())
			{
				tripletList.push_back(T(it->second, it_neighb->second, 1.0f));
			}

		}
		mat_b_right.setZero();
		mat_b_right.setFromTriplets(tripletList.begin(), tripletList.end());

		mat_blur += mat_b_left;
		mat_blur += mat_b_right;
	}

	std::chrono::steady_clock::time_point end_blur_construction = std::chrono::steady_clock::now();
	std::cout << "blur construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_blur_construction - begin_blur_construction).count() << "ms" << std::endl;

	mat_slice.finalize();
	mat_splat.finalize();
	mat_blur.finalize();
	//mat_slice.makeCompressed();
	//mat_splat.makeCompressed();
	//mat_blur.makeCompressed();

	
	//
	Eigen::MatrixXf mat_ones(w*h, 1);
	mat_ones.setOnes();
	Eigen::MatrixXf mat_ones_splatted = mat_splat * mat_ones;
	mat_normalizer = mat_slice * mat_ones_splatted;

}

Eigen::MatrixXf bilateral_grid_simplified::blur(Eigen::MatrixXf& in)
{
	return  mat_blur * in + (in *(5.0f * 2.0f));
}


Eigen::SparseMatrix<float, Eigen::RowMajor> bilateral_grid_simplified::blur(const Eigen::SparseMatrix<float, Eigen::RowMajor>& in)
{
	Eigen::SparseMatrix<float, Eigen::RowMajor> a = (in *(5.0f * 2.0f));
	Eigen::SparseMatrix<float, Eigen::RowMajor> b = mat_blur * in;
	return b + a;
}

cv::Mat bilateral_grid_simplified::filter(cv::Mat input_image)
{
	std::chrono::steady_clock::time_point start_blur = std::chrono::steady_clock::now();

	Eigen::Map<Eigen::MatrixXf> eig_input_image(reinterpret_cast<float*>(input_image.data), input_image.cols * input_image.rows, 1);
	Eigen::MatrixXf mat_splatted = mat_splat * eig_input_image;
	Eigen::MatrixXf mat_splatted_and_blurred = blur(mat_splatted);
	Eigen::MatrixXf mat_blurred = mat_slice * mat_splatted_and_blurred;

	Eigen::MatrixXf mat_blurred_result = mat_blurred.cwiseQuotient(mat_normalizer);

	cv::Mat cv_blurred_result(input_image.rows, input_image.cols, CV_32FC1);
	memcpy(cv_blurred_result.data, mat_blurred_result.data(), input_image.rows * input_image.cols * sizeof(float));

	std::chrono::steady_clock::time_point end_blur = std::chrono::steady_clock::now();
	std::cout << "blur: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_blur - start_blur).count() << "ms" << std::endl;

	return cv_blurred_result;
}

cv::Mat bilateral_grid_simplified::splat_slice(cv::Mat input_image)
{
	Eigen::Map<Eigen::MatrixXf> eig_input_image(reinterpret_cast<float*>(input_image.data), input_image.cols * input_image.rows, 1);
	Eigen::MatrixXf mat_splatted = mat_splat * eig_input_image;
	Eigen::MatrixXf mat_sliced = mat_slice * mat_splatted;

	Eigen::MatrixXf result = mat_sliced.cwiseQuotient(mat_normalizer);

	cv::Mat cv_result(input_image.rows, input_image.cols, CV_32FC1);
	memcpy(cv_result.data, result.data(), input_image.rows * input_image.cols * 4);

	return cv_result;
}



//void serialize(std::string directory)
//{
//	//Serialize(directory + "\\mat_splat.dat", mat_splat);
//	//Serialize(directory + "\\mat_slice.dat", mat_slice);
//	//Serialize(directory + "\\mat_blur.dat", mat_blur);
//}
//
//void deserialize(std::string directory)
//{
//	/*	std::chrono::steady_clock::time_point begin_deserialize = std::chrono::steady_clock::now();
//
//	Deserialize(directory + "\\mat_splat.dat", mat_splat);
//	Deserialize(directory + "\\mat_slice.dat", mat_slice);
//	Deserialize(directory + "\\mat_blur.dat", mat_blur);
//	nb_vertices = static_cast<int>(mat_splat.outerSize());
//
//	std::chrono::steady_clock::time_point end_deserialize = std::chrono::steady_clock::now();
//	std::cout << "bilateral_grid::deserialize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_deserialize - begin_deserialize).count() << "ms" << std::endl;
//	*/
//}