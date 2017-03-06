/* Copyright 2017 Toon Van den Zegel. All Rights Reserved.                              */
/*                                                                                      */
/* This file is part of fast_bilateral_space_stereo.                                    */
/*                                                                                      */
/* fast_bilateral_space_stereo is free software :                                       */
/* you can redistribute it and / or modify                                              */
/* it under the terms of the GNU General Public License as published by                 */
/* the Free Software Foundation, either version 3 of the License, or                    */
/* (at your option) any later version.                                                  */
/*                                                                                      */
/* fast_bilateral_space_stereo is distributed in the hope that it will be useful,       */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of                       */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the                          */
/* GNU General Public License for more details.                                         */
/*                                                                                      */
/* You should have received a copy of the GNU General Public License                    */
/* along with fast_bilateral_space_stereo.                                              */
/* If not, see <http://www.gnu.org/licenses/>.                                          */

#include "stereo_matcher_birchfield_tomasi.h"

stereo_matcher_birchfield_tomasi::stereo_matcher_birchfield_tomasi()
{

}

stereo_matcher_birchfield_tomasi::~stereo_matcher_birchfield_tomasi()
{

}

void stereo_matcher_birchfield_tomasi::stereo_match(cv::Mat stereo_images[2])
{
	assert(stereo_images[0].type() == CV_8UC1); // it should be grayscale
	assert(stereo_images[1].type() == CV_8UC1);

	const int disparity_min = current_parameters.disparity_min;
	const int disparity_max = current_parameters.disparity_max;
	const int noise_epsilon = current_parameters.noise_epsilon;

	std::chrono::steady_clock::time_point begin_stereo_match_construction = std::chrono::steady_clock::now();

	// a small blur
	cv::Mat stereo_filt_images[2];
	cv::boxFilter(stereo_images[0], stereo_filt_images[0], -1, cv::Size(2, 2));
	cv::boxFilter(stereo_images[1], stereo_filt_images[1], -1, cv::Size(2, 2));

	// min-max kernel
	cv::Mat stereo_images_upper[2];
	cv::Mat stereo_images_lower[2];
	cv::Mat minmax_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	for (int i = 0; i < 2; ++i)
	{
		cv::erode(stereo_filt_images[i], stereo_images_lower[i], minmax_kernel);
		stereo_images_lower[i] -= noise_epsilon;
		cv::dilate(stereo_filt_images[i], stereo_images_upper[i], minmax_kernel);
		stereo_images_upper[i] += noise_epsilon;
	}

	const int width = stereo_images[0].cols;
	const int height = stereo_images[0].rows;

	// a lot of temporary images, makes debugging more easy :)
	cv::Mat& block_match_image = current_output.block_match_image;
	cv::Mat& block_match_image1_x = current_output.block_match_image1_x;
	cv::Mat& block_match_image2_x = current_output.block_match_image2_x;
	cv::Mat& block_match_image1_y = current_output.block_match_image1_y;
	cv::Mat& block_match_image_final = current_output.block_match_image_final;

	block_match_image.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);
	block_match_image1_x.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);
	block_match_image2_x.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);
	block_match_image1_y.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);
	block_match_image_final.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);

	cv::Mat& min_disp_image = current_output.min_disp_image;
	cv::Mat& max_disp_image = current_output.max_disp_image;
	min_disp_image.create(stereo_images[0].rows, stereo_images[0].cols, CV_16SC1);
	max_disp_image.create(stereo_images[0].rows, stereo_images[0].cols, CV_16SC1);
	min_disp_image = cv::Scalar(std::numeric_limits<int16_t>::max());
	max_disp_image = cv::Scalar(-std::numeric_limits<int16_t>::max());

	for (int d = disparity_min; d <= disparity_max; d += 1)
	{
		// check upper and lower bounds in order to see if we have a match
		// at this certain disparity
		if (d < 0) // negative disparity
		{
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < -d; ++x)
				{
					int idx = x + y * width;
					block_match_image.data[idx] = 0;
				}

				for (int x = -d; x < width; ++x)
				{
					int idx = x + y * width;
					block_match_image.data[idx] =
						(stereo_images_upper[0].data[idx] >= stereo_images_lower[1].data[idx + d])
						&&
						(stereo_images_lower[0].data[idx] <= stereo_images_upper[1].data[idx + d]);
				}
			}
		}
		else // positive disparity
		{
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width - d; ++x)
				{
					int idx = x + y * width;
					block_match_image.data[idx] =
						(stereo_images_upper[0].data[idx] >= stereo_images_lower[1].data[idx + d])
						&&
						(stereo_images_lower[0].data[idx] <= stereo_images_upper[1].data[idx + d]);
				}


				for (int x = width - d; x < width; ++x)
				{
					int idx = x + y * width;
					block_match_image.data[idx] = 0;
				}
			}
		}

		// execute an erosion filter in order to supress the noise
		if (current_parameters.filter_size == block_filter_size::size_5x5)
		{
			block_filter_horz_21012(block_match_image, block_match_image1_x);
			block_filter_vert_21012(block_match_image1_x, block_match_image_final);
		}
		else if (current_parameters.filter_size == block_filter_size::size_15x15)
		{
			block_filter_horz_21012(block_match_image, block_match_image1_x);
			block_filter_horz_505(block_match_image1_x, block_match_image2_x);

			block_filter_vert_21012(block_match_image2_x, block_match_image1_y);
			block_filter_vert_505(block_match_image1_y, block_match_image_final);
		}
		else if (current_parameters.filter_size == block_filter_size::size_25x25)
		{
			block_filter_horz_21012(block_match_image, block_match_image1_x);
			block_filter_horz_1050510(block_match_image1_x, block_match_image2_x);

			block_filter_vert_21012(block_match_image2_x, block_match_image1_y);
			block_filter_vert_1050510(block_match_image1_y, block_match_image_final);
		}


		for (int i = 0; i < width * height; ++i)
		{
			if (block_match_image_final.data[i])
			{
				if (min_disp_image.at<int16_t>(i) == std::numeric_limits<int16_t>::max())
				{
					min_disp_image.at<int16_t>(i) = std::min(min_disp_image.at<int16_t>(i), (int16_t)d);
				}

				max_disp_image.at<int16_t>(i) = std::max(max_disp_image.at<int16_t>(i), (int16_t)d);
			}
		}


	}

	cv::Mat& conf_disp_image = current_output.conf_disp_image;
	conf_disp_image.create(stereo_images[0].rows, stereo_images[0].cols, CV_8UC1);
	for (int i = 0; i < width * height; ++i)
	{
		if (min_disp_image.at<int16_t>(i) == std::numeric_limits<int16_t>::max())
		{
			min_disp_image.at<int16_t>(i) = disparity_min;
			max_disp_image.at<int16_t>(i) = disparity_max;
			//	min_disp_image.at<int16_t>(i) = std::numeric_limits<int16_t>::max();
			//	max_disp_image.at<int16_t>(i) = std::numeric_limits<int16_t>::max();

			conf_disp_image.data[i] = 0;
		}
		else
		{
			conf_disp_image.data[i] = 255;
		}
	}

	std::chrono::steady_clock::time_point end_stereo_match_construction = std::chrono::steady_clock::now();
	std::cout << "stereo match: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_stereo_match_construction - begin_stereo_match_construction).count() << "ms" << std::endl;
}

void stereo_matcher_birchfield_tomasi::generate_data_loss_table(const bilateral_grid_simplified& grid, std::vector<int>& lookup)
{
	const int nb_vertices = grid.get_nb_vertices();
	const int disparity_min = static_cast<int>(current_parameters.disparity_min);
	const int disparity_max = static_cast<int>(current_parameters.disparity_max);
	const int noise_epsilon = static_cast<int>(current_parameters.noise_epsilon);
	const int disparity_range = static_cast<int>((disparity_max - disparity_min) + 1);
	const cv::Mat& max_disp_image = current_output.max_disp_image;
	const cv::Mat& min_disp_image = current_output.min_disp_image;

	std::chrono::steady_clock::time_point begin_data_loss = std::chrono::steady_clock::now();

	lookup.clear();
	lookup.resize(nb_vertices * disparity_range, 0);

	for (int vertex_id = 0, vertex_id_end = nb_vertices; vertex_id < vertex_id_end; ++vertex_id)
	{
		int counter = 0;
		int* plookup = &lookup[vertex_id * disparity_range];
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(grid.get_splat_matrix(), vertex_id); it; ++it) // loop through pixels of that vertex
		{
			const int pixel_id = it.index();
			const int pixel_weight = static_cast<int>(it.value());

			int gj = 0;
			for (int j = max_disp_image.at<int16_t>(pixel_id) + 1 - disparity_min; j < disparity_range; ++j)
			{
				gj += pixel_weight;
				plookup[j] += gj;
			}

			gj = 0;
			for (int j = (int)min_disp_image.at<int16_t>(pixel_id) - 1 - disparity_min; j >= 0; --j)
			{
				gj += pixel_weight;
				plookup[j] += gj;
			}

		}

	}


	std::chrono::steady_clock::time_point end_data_loss = std::chrono::steady_clock::now();
	std::cout << "data loss: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_data_loss - begin_data_loss).count() << "ms" << std::endl;

	// debug
	// cv::Mat debug_lookup_image(nb_vertices, matcher.disparty_max + 1, CV_32SC1);
	// memcpy(debug_lookup_image.data, lookup.data(), lookup.size() * sizeof(int));

}

void stereo_matcher_birchfield_tomasi::block_filter_horz_21012(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;

	int idx = 0;
	for (int y = 0; y < height; ++y)
	{
		out.data[idx] = in.data[idx] &&
			in.data[idx + 1] && in.data[idx + 2];
		++idx;

		out.data[idx] = in.data[idx] && in.data[idx - 1] &&
			in.data[idx + 1] && in.data[idx + 2];
		++idx;

		for (int x = 2; x < width - 2; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 1] && in.data[idx - 2] &&
				in.data[idx + 1] && in.data[idx + 2];
			++idx;
		}

		out.data[idx] = in.data[idx] && in.data[idx - 1] &&
			in.data[idx - 2] && in.data[idx + 1];
		++idx;

		out.data[idx] = in.data[idx] && in.data[idx - 1] && in.data[idx - 2];
		++idx;
	}
}

void stereo_matcher_birchfield_tomasi::block_filter_horz_1050510(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;

	int idx = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < 5; ++x)
		{
			out.data[idx] = in.data[idx] &&
				in.data[idx + 5] && in.data[idx + 10];
			++idx;
		}

		for (int x = 5; x < 10; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5] &&
				in.data[idx + 5] && in.data[idx + 10];
			++idx;
		}

		for (int x = 10; x < width - 10; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5] && in.data[idx - 10] &&
				in.data[idx + 5] && in.data[idx + 10];
			++idx;
		}

		for (int x = width - 10; x < width - 5; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5] && in.data[idx - 10] &&
				in.data[idx + 5];
			++idx;
		}

		for (int x = width - 5; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5] && in.data[idx - 10];
			++idx;
		}
	}
}

void stereo_matcher_birchfield_tomasi::block_filter_horz_505(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;

	int idx = 0;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < 5; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx + 5];
			++idx;
		}

		for (int x = 5; x < width - 5; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5] && in.data[idx + 5];
			++idx;
		}

		for (int x = width - 5; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - 5];
			++idx;
		}
	}
}

void stereo_matcher_birchfield_tomasi::block_filter_vert_21012(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;
	const int stride_1 = 1 * width;
	const int stride_2 = 2 * width;

	int idx = 0;
	for (int x = 0; x < width; ++x)
	{
		out.data[idx] = in.data[idx] &&
			in.data[idx + stride_1] && in.data[idx + stride_2];
		++idx;
	}

	for (int x = 0; x < width; ++x)
	{
		out.data[idx] = in.data[idx] && in.data[idx - stride_1] &&
			in.data[idx + stride_1] && in.data[idx + stride_2];
		++idx;
	}

	for (int y = 2; y < height - 2; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx = x + y * width;

			out.data[idx] = in.data[idx] && in.data[idx - stride_1] && in.data[idx - stride_2] &&
				in.data[idx + stride_1] && in.data[idx + stride_2];
		}
	}

	for (int x = 0; x < width; ++x)
	{
		out.data[idx] = in.data[idx] && in.data[idx - stride_1] && in.data[idx - stride_2] &&
			in.data[idx + stride_1];
		++idx;
	}

	for (int x = 0; x < width; ++x)
	{
		out.data[idx] = in.data[idx] && in.data[idx - stride_1] && in.data[idx - stride_2];
		++idx;
	}

}

void stereo_matcher_birchfield_tomasi::stereo_matcher_birchfield_tomasi::block_filter_vert_1050510(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;
	const int stride_5 = 5 * width;
	const int stride_10 = 10 * width;

	int idx = 0;
	for (int y = 0; y < 5; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] &&
				in.data[idx + stride_5] && in.data[idx + stride_10];
			++idx;
		}
	}

	for (int y = 5; y < 10; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - stride_5] &&
				in.data[idx + stride_5] && in.data[idx + stride_10];
			++idx;
		}
	}

	for (int y = 10; y < height - 10; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx = x + y * width;

			out.data[idx] = in.data[idx] && in.data[idx - stride_5] && in.data[idx - stride_10] &&
				in.data[idx + stride_5] && in.data[idx + stride_10];
		}
	}

	for (int y = height - 10; y < height - 5; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - stride_5] && in.data[idx - stride_10] &&
				in.data[idx + stride_5];
			++idx;
		}
	}

	for (int y = height - 5; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - stride_5] && in.data[idx - stride_10];
			++idx;
		}
	}
}

void stereo_matcher_birchfield_tomasi::stereo_matcher_birchfield_tomasi::block_filter_vert_505(const cv::Mat& in, cv::Mat& out)
{
	assert(in.type() == out.type());

	const int width = in.cols;
	const int height = in.rows;
	const int stride_5 = 5 * width;

	int idx = 0;
	for (int y = 0; y < 5; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] &&
				in.data[idx + stride_5];
			++idx;
		}
	}

	for (int y = 5; y < height - 5; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx = x + y * width;

			out.data[idx] = in.data[idx] && in.data[idx - stride_5] && in.data[idx + stride_5];
		}
	}

	for (int y = height - 5; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			out.data[idx] = in.data[idx] && in.data[idx - stride_5];
			++idx;
		}
	}
}

//void serialize(std::string directory)
//{
//	cv::imwrite(directory + "\\min_disp_image.png", min_disp_image);
//	cv::imwrite(directory + "\\max_disp_image.png", max_disp_image);
//	cv::imwrite(directory + "\\conf_disp_image.png", conf_disp_image);
//}

//void deserialize(std::string directory)
//{
//	std::chrono::steady_clock::time_point begin_deserialize = std::chrono::steady_clock::now();

//	min_disp_image = cv::imread(directory + "\\min_disp_image.png", CV_LOAD_IMAGE_GRAYSCALE);
//	max_disp_image = cv::imread(directory + "\\max_disp_image.png", CV_LOAD_IMAGE_GRAYSCALE);
//	conf_disp_image = cv::imread(directory + "\\conf_disp_image.png", CV_LOAD_IMAGE_GRAYSCALE);

//	make_float_images();

//	std::chrono::steady_clock::time_point end_deserialize = std::chrono::steady_clock::now();
//	std::cout << "stereo_matcher::deserialize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_deserialize - begin_deserialize).count() << "ms" << std::endl;

//}