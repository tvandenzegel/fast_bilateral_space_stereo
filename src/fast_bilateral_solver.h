#pragma once

#include <Eigen/Sparse>
#include "eigen_sparse_serialize.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "glog/logging.h"
#include "ceres/ceres.h"

#include <string>
#include <chrono>
#include <unordered_map>
#include <cassert>

#include "bilateral_grid_simplified.h"

class fast_bilateral_solver
{
	class fast_bilateral_problem : public ceres::FirstOrderFunction
	{
	public:
		fast_bilateral_problem(
			const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat_C_CBC,
			const bilateral_grid_simplified& grid,
			const std::vector<int>& lookup,
			int disparty_range,
			float lambda):
			mat_C_CBC(mat_C_CBC),
			grid(grid),
			lookup(lookup),
			disparty_range(disparty_range),
			lambda(lambda),
			nb_vertices(grid.get_nb_vertices())
		{
			intermediate_prev_energy = std::numeric_limits<double>::max();
		}
		
		const Eigen::SparseMatrix<float, Eigen::RowMajor>& mat_C_CBC;
		const bilateral_grid_simplified& grid;
		const std::vector<int>& lookup;
		const int disparty_range;
		const float lambda;
		const int nb_vertices;

		mutable double intermediate_prev_energy;
		mutable std::vector<Eigen::VectorXf> intermediate_results;

		virtual bool Evaluate(const double* x,
			double* cost,
			double* grad) const
		{
			Eigen::VectorXf x_eig = Eigen::VectorXf(nb_vertices);
			Eigen::VectorXf grad_eig = Eigen::VectorXf(nb_vertices);

			
			for (int i = 0; i < nb_vertices; ++i)
			{
				x_eig(i) = x[i];
			}

			float loss_smoothing_term = x_eig.transpose() *	mat_C_CBC * x_eig;




			grad_eig = (mat_C_CBC * x_eig) * 2.0f;

			for (int i = 0; i < nb_vertices; ++i)
			{
				grad[i] = grad_eig(i);
			}

			/// Data term
			const float fdisparty_max = (float)disparty_range;
			float loss_data_term = 0.f;
			for (size_t j = 0; j < nb_vertices; ++j)
			{

				float vj = x[j];

				float grad_data = 0.f;
				if (vj < 0.f)
				{
					grad_data = -(500.0f - 0);
					loss_data_term += 500.0f;
				}
				else if (vj >= fdisparty_max)
				{
					grad_data = -(0.f - 500.0f);
					loss_data_term += 500.0f;
				}
				else
				{
					int vj_id = (int)std::floor(vj);
					float gj_floor = lookup[j * disparty_range + vj_id];
					float gj_ceil = lookup[j * disparty_range + (vj_id + 1)];

					grad_data = -(gj_floor - gj_ceil);

					float v1 = ((float)(vj_id + 1) - vj) * gj_floor;
					float v2 = (vj - (float)vj_id) * gj_ceil;
					loss_data_term += v1 + v2;


				}
				grad[j] += grad_data *lambda;

			}



			///////
			float loss_total = loss_smoothing_term + lambda * loss_data_term;
			cost[0] = loss_total;

			if ((loss_total < intermediate_prev_energy) && (intermediate_results.size() < 100))
			{
				intermediate_results.push_back(x_eig);
				intermediate_prev_energy = loss_total;
			}
				


			return true;
		}





		virtual int NumParameters() const { return nb_vertices; }
	};


public:
	fast_bilateral_solver()
	{
		//google::InitGoogleLogging(nullptr);
	}

	std::vector<cv::Mat> intermediate_results;
	

	void bistochastize(bilateral_grid_simplified& grid, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dn, Eigen::SparseMatrix<float, Eigen::RowMajor>& Dm, const int nb_iterations = 16)
	{
		Eigen::MatrixXf mat_pixs(grid.get_splat_matrix().innerSize(), 1);
		mat_pixs.setOnes();
		Eigen::MatrixXf mat_m = grid.get_splat_matrix() * mat_pixs;

		Eigen::MatrixXf mat_n(grid.get_nb_vertices(), 1);
		mat_n.setOnes();

		Eigen::MatrixXf mat_n_new;
		for (int i = 0; i < nb_iterations; ++i)
		{

			Eigen::MatrixXf ress = (mat_n.cwiseProduct(mat_m)).cwiseQuotient(grid.blur(mat_n));

			mat_n_new = ress.cwiseSqrt();

			Eigen::MatrixXf diff = mat_n_new - mat_n;
			float dd = diff.sum();
			std::cout << dd << std::endl;
			mat_n = mat_n_new;
		}
		mat_m = mat_n.cwiseProduct(grid.blur(mat_n));

		Dm = Eigen::SparseMatrix<float, Eigen::RowMajor>(grid.get_nb_vertices(), grid.get_nb_vertices());
		{
			typedef Eigen::Triplet<float> T;
			std::vector<T> tripletList;
			for (size_t i = 0; i < grid.get_nb_vertices(); ++i)
			{
				tripletList.push_back(T(i, i, mat_m(i, 0)));

			}
			Dm.setFromTriplets(tripletList.begin(), tripletList.end());
		}

		Dn = Eigen::SparseMatrix<float, Eigen::RowMajor>(grid.get_nb_vertices(), grid.get_nb_vertices());
		{
			typedef Eigen::Triplet<float> T;
			std::vector<T> tripletList;
			for (size_t i = 0; i < grid.get_nb_vertices(); ++i)
			{
				tripletList.push_back(T(i, i, mat_n.coeffRef(i, 0)));

			}
			Dn.setFromTriplets(tripletList.begin(), tripletList.end());
		}

		

	}




	cv::Mat solve(cv::Mat initial_image_float, bilateral_grid_simplified& grid, const std::vector<int>& lookup, const int disparity_range, const float lambda, const int max_num_iterations = 25)
	{
		assert(initial_image_float.type() == CV_32FC1);

		const int nb_pixels = initial_image_float.cols * initial_image_float.rows;

		/// fill in initial parameters/image
		double* parameters = new double[grid.get_nb_vertices()];
		splat_initial_image(grid, initial_image_float, parameters);
		
		// create smoothing matrix
		Eigen::SparseMatrix<float, Eigen::RowMajor> Dn, Dm;
		bistochastize(grid, Dn, Dm);
		const Eigen::SparseMatrix<float, Eigen::RowMajor> mat_CBC = Dn * grid.blur(Dn);
		const Eigen::SparseMatrix<float, Eigen::RowMajor> mat_C_CBC = Dm - mat_CBC;
		
		// setup solver & solve
		ceres::GradientProblemSolver::Options options;
		options.minimizer_progress_to_stdout = true;
		options.max_num_iterations = max_num_iterations;
		ceres::GradientProblemSolver::Summary summary;
		fast_bilateral_problem* pr = new fast_bilateral_problem(mat_C_CBC, grid, lookup, disparity_range, lambda);
		ceres::GradientProblem problem(pr);
		ceres::Solve(options, problem, parameters, &summary);

		std::cout << summary.FullReport() << "\n";

		// copy result in to eigen vector
		Eigen::VectorXf eig_parameters(grid.get_nb_vertices());
		for (int i = 0; i < grid.get_nb_vertices(); ++i)
		{
			eig_parameters(i) = parameters[i];
		}

		// slice the result!
		Eigen::MatrixXf eig_result = grid.get_slice_matrix() * eig_parameters;

		// to and opencv image
		cv::Mat result_image(initial_image_float.rows, initial_image_float.cols, CV_32FC1);
		memcpy(result_image.data, eig_result.data(), sizeof(float) * nb_pixels);

		// process intermediate results for debugging
		{
			intermediate_results.clear();
			intermediate_results.resize(pr->intermediate_results.size());
			Eigen::MatrixXf eig_result2;
			for (size_t i = 0; i < pr->intermediate_results.size(); ++i)
			{
				eig_result2 = grid.get_slice_matrix() * pr->intermediate_results[i];
				auto& img = intermediate_results[i];
				img.create(initial_image_float.rows, initial_image_float.cols, CV_32FC1);
				memcpy(img.data, eig_result2.data(), sizeof(float) * nb_pixels);
			}

		}
	

		delete[] parameters;

		return result_image;
	}

	void splat_initial_image(bilateral_grid_simplified& grid, cv::Mat initial_image_float, double* parameters)
	{
		assert(initial_image_float.type() == CV_32FC1);
		const int nb_pixels = initial_image_float.cols * initial_image_float.rows;
		Eigen::VectorXf in_vec(nb_pixels);
		const float *pff = (const float*)initial_image_float.data;
		for (int i = 0; i < nb_pixels; ++i)
		{
			in_vec(i) = pff[i];
		}

		// splat input image
		Eigen::VectorXf splat_in_vec = grid.get_splat_matrix() * in_vec;

		// the grid is not normalized, so splat a bunch of 'ones' so we can normalize the result
		Eigen::MatrixXf normalization_vec(nb_pixels, 1);
		normalization_vec.setOnes();
		Eigen::MatrixXf normalization_weights = grid.get_splat_matrix() * normalization_vec;

		splat_in_vec = splat_in_vec.cwiseQuotient(normalization_weights);

		for (int i = 0; i < grid.get_nb_vertices(); ++i)
		{
			parameters[i] = splat_in_vec(i);
		}

	}
	


};