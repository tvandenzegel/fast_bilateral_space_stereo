#pragma once

#include "bilateral_grid_simplified.h"

#include <Eigen/Sparse>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <chrono>

class stereo_matcher_birchfield_tomasi
{
public:
	enum class block_filter_size
	{
		size_5x5,
		size_25x25
	};

	struct parameters
	{
		parameters():
			disparity_min(0),
			disparity_max(16),
			noise_epsilon(4),
			filter_size(block_filter_size::size_25x25)
		{

		}

		int disparity_min;
		int disparity_max;
		int noise_epsilon;
		block_filter_size filter_size;
	};
	parameters& get_parameters() { return current_parameters; }

	struct output
	{
		cv::Mat min_disp_image;
		cv::Mat max_disp_image;
		cv::Mat conf_disp_image;

		// Debugging images
		cv::Mat block_match_image;
		cv::Mat block_match_image1_x;
		cv::Mat block_match_image2_x;
		cv::Mat block_match_image1_y;
		cv::Mat block_match_image_final;
	};
	const output& get_output() { return current_output; }

public:
	stereo_matcher_birchfield_tomasi();
	~stereo_matcher_birchfield_tomasi();

	output current_output;
	parameters current_parameters;
	
	void stereo_match(cv::Mat stereo_images[2]);

	void generate_data_loss_table(const bilateral_grid_simplified& grid, std::vector<int>& lookup);
	

private:


	void block_filter_horz_21012(const cv::Mat& in, cv::Mat& out);
	void block_filter_horz_1050510(const cv::Mat& in, cv::Mat& out);
	
	void block_filter_vert_21012(const cv::Mat& in, cv::Mat& out);
	void block_filter_vert_1050105(const cv::Mat& in, cv::Mat& out);
};