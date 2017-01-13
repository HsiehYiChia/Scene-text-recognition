#include "ER.h"


// ====================================================
// ======================== ER ========================
// ====================================================
ER::ER(const int level_, const int pixel_, const int x_, const int y_) : level(level_), pixel(pixel_), area(1), done(false), stability(.0), 
																		parent(nullptr), child(nullptr), next(nullptr)
{
	bound = Rect(x_, y_, 1, 1);
}

// ====================================================
// ===================== ER_filter ====================
// ====================================================
ERFilter::ERFilter(int thresh_step, int min_area, int max_area, int stability_t, double overlap_coef) : THRESH_STEP(thresh_step), MIN_AREA(min_area), MAX_AREA(max_area),
																										STABILITY_T(stability_t), OVERLAP_COEF(overlap_coef)
{

}


inline void ERFilter::er_accumulate(ER *er, const int &current_pixel, const int &x, const int &y)
{
	er->area++;

	const int x1 = min(er->bound.x, x);
	const int x2 = max(er->bound.br().x - 1, x);
	const int y1 = min(er->bound.y, y);
	const int y2 = max(er->bound.br().y - 1, y);

	er->bound.x = x1;
	er->bound.y = y1;
	er->bound.width = x2 - x1 + 1;
	er->bound.height = y2 - y1 + 1;
}

void ERFilter::er_merge(ER *parent, ER *child)
{
	parent->area += child->area;

	const int x1 = min(parent->bound.x, child->bound.x);
	const int x2 = max(parent->bound.br().x - 1, child->bound.br().x - 1);
	const int y1 = min(parent->bound.y, child->bound.y);
	const int y2 = max(parent->bound.br().y - 1, child->bound.br().y - 1);

	parent->bound.x = x1;
	parent->bound.y = y1;
	parent->bound.width = x2 - x1 + 1;
	parent->bound.height = y2 - y1 + 1;

	if (child->area <= MIN_AREA ||
		child->bound.height <= 2 ||
		child->bound.width <= 2)
	{
		ER *new_child = child->child;

		if (new_child)
		{
			while (new_child->next)
				new_child = new_child->next;
			new_child->next = parent->child;
			parent->child = child->child;
			child->child->parent = parent;
		}
		delete child;
	}
	else
	{
		child->next = parent->child;
		parent->child = child;
		child->parent = parent;
	}
}


void ERFilter::er_save(ER *er)
{
	// Non Recursive Preorder Tree Traversal
	// See http://algorithms.tutorialhorizon.com/binary-tree-preorder-traversal-non-recursive-approach/ for more info.

	// 1. Create a Stack
	vector<ER *> tree_stack;
	ER *root = er;

save_step_2:
	// 2. Print the root and push it to Stack and go left, i.e root=root.left and till it hits the nullptr.
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
	}

	// 3. If root is null and Stack is empty Then
	//		return, we are done.
	if (root == nullptr && tree_stack.empty())
	{
		return;
	}

	// 4. Else
	//		Pop the top Node from the Stack and set it as, root = popped_Node.
	//		Go right, root = root.right.
	//		Go to step 2.

	root = tree_stack.back();
	tree_stack.pop_back();
	root = root->next;
	goto save_step_2;

	// 5. End If
}


// extract the component tree and store all the ER regions
// base on OpenCV source code, see https://github.com/Itseez/opencv_contrib/tree/master/modules/text for more info
// uses the algorithm described in 
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
ER* ERFilter::er_tree_extract(Mat input)
{
	CV_Assert(input.type() == CV_8UC1);

	Mat input_clone = input.clone();
	const int width = input_clone.cols;
	const int height = input_clone.rows;
	const int highest_level = (255 / THRESH_STEP) + 1;
	const uchar *imgData = input_clone.data;

	input_clone /= THRESH_STEP;

	//!< 1. Clear the accessible pixel mask, the heap of boundary pixels and the component
	bool *pixel_accessible = new bool[height*width]();
	bool *pixel_accumulated = new bool[height*width]();
	vector<int> boundary_pixel[256];
	vector<int> boundary_edge[256];
	vector<ER *>er_stack;
	
	int priority = highest_level;


	//!< 1-2. push a dummy-component onto the stack, 
	//!<	  with grey-level heigher than any allowed in the image
	er_stack.push_back(new ER(256, 0, 0, 0));


	//!< 2. make the top-right corner the source pixel, get its gray level and mark it accessible
	int current_pixel = 0;
	int current_edge = 0;
	int current_level = imgData[current_pixel];
	pixel_accessible[current_pixel] = true;

	
step_3:
	int x = current_pixel % width;
	int y = current_pixel / width;

	//!< 3. push an empty component with current_level onto the component stack
	er_stack.push_back(new ER(current_level, current_pixel, x, y));


	for (;;)
	{
		//!< 4. Explore the remaining edges to the neighbors of the current pixel, in order, as follows : 
		//!<	For each neighbor, check if the neighbor is already accessible.If it
		//!<	is not, mark it as accessible and retrieve its grey - level.If the grey - level is not
		//!<	lower than the current one, push it onto the heap of boundary pixels.If on
		//!<	the other hand the grey - level is lower than the current one, enter the current
		//!<	pixel back into the queue of boundary pixels for later processing(with the
		//!<	next edge number), consider the new pixel and its grey - level and go to 3.
		int neighbor_pixel;
		int neighbor_level;
		

		for (; current_edge < 4; current_edge++)
		{
			switch (current_edge)
			{
				case right	: neighbor_pixel = (x + 1 < width)	? current_pixel + 1		: current_pixel;	break;
				case bottom	: neighbor_pixel = (y + 1 < height) ? current_pixel + width : current_pixel;	break;
				case left	: neighbor_pixel = (x > 0)			? current_pixel - 1		: current_pixel;	break;
				case top	: neighbor_pixel = (y > 0)			? current_pixel - width : current_pixel;	break;
				default: break;
			}
						
			if (!pixel_accessible[neighbor_pixel] && neighbor_pixel != current_pixel)
			{
				pixel_accessible[neighbor_pixel] = true;
				neighbor_level = imgData[neighbor_pixel];

				if (neighbor_level >= current_level)
				{
					boundary_pixel[neighbor_level].push_back(neighbor_pixel);
					boundary_edge[neighbor_level].push_back(0);

					if (neighbor_level < priority)
						priority = neighbor_level;
				}
				else
				{
					boundary_pixel[current_level].push_back(current_pixel);
					boundary_edge[current_level].push_back(current_edge + 1);

					if (current_level < priority)
						priority = current_level;

					current_pixel = neighbor_pixel;
					current_level = neighbor_level;
					current_edge = 0;
					goto step_3;
				}
			}
		}




		//!< 5. Accumulate the current pixel to the component at the top of the stack 
		//!<	(water saturates the current pixel).
		er_accumulate(er_stack.back(), current_pixel, x, y);
		pixel_accumulated[current_pixel] = true;

		//!< 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the
		//!<	returned pixel is at the same grey - level as the previous, go to 4	
		if (priority == highest_level)
		{
			// In er_save, local maxima ERs in first stage will be save to pool
			//er_save(er_stack.back());
			
			
			delete[] pixel_accessible;
			delete[] pixel_accumulated;
			return er_stack.back();
		}
			
			
		int new_pixel = boundary_pixel[priority].back();
		int new_edge = boundary_edge[priority].back();
		int new_pixel_grey_level = imgData[new_pixel];

		boundary_pixel[priority].pop_back();
		boundary_edge[priority].pop_back();

		while (boundary_pixel[priority].empty() && priority < highest_level)
			priority++;

		current_pixel =  new_pixel;
		current_edge = new_edge;
		x = current_pixel % width;
		y = current_pixel / width;

		if (new_pixel_grey_level != current_level)
		{
			//!< 7. The returned pixel is at a higher grey-level, so we must now process all
			//!<	components on the component stack until we reach the higher grey - level.
			//!<	This is done with the ProcessStack sub - routine, see below.Then go to 4.
			current_level = new_pixel_grey_level;
			process_stack(new_pixel_grey_level, er_stack);
		}
	}
}


void ERFilter::process_stack(const int new_pixel_grey_level, vector<ER *> &er_stack)
{
	do
	{
		//!< 1. Process component on the top of the stack. The next grey-level is the minimum
		//!<	of new_pixel_grey_level and the grey - level for the second component on
		//!<	the stack.
		ER *top = er_stack.back();
		er_stack.pop_back();
		ER *second_top = er_stack.back();

		//!< 2. If new_pixel_grey_level is smaller than the grey - level on the second component
		//!<	on the stack, set the top of stack grey - level to new_pixel_grey_level and return
		//!<	from sub - routine(This occurs when the new pixel is at a grey - level for which
		//!<	there is not yet a component instantiated, so we let the top of stack be that
		//!<	level by just changing its grey - level.
		if (new_pixel_grey_level < second_top->level)
		{
			top->level = new_pixel_grey_level;
			er_stack.push_back(top);
			return;
		}

		//!< 3. Remove the top of stack and merge it into the second component on stack
		//!<	as follows : Add the first and second moment accumulators together and / or
		//!<	join the pixel lists.Either merge the histories of the components, or take the
		//!<	history from the winner.Note here that the top of stack should be considered
		//!<	one ’time - step’ back, so its current size is part of the history.Therefore the
		//!<	top of stack would be the winner if its current size is larger than the previous
		//!<	size of second on stack.
		er_merge(second_top, top);
		
	}
	//!< 4. If(new_pixel_grey_level>top_of_stack_grey_level) go to 1.
	while (new_pixel_grey_level > er_stack.back()->level);
}

void ERFilter::non_maximum_supression(ER *er, ERs &pool, Mat input)
{
	// Non Recursive Preorder Tree Traversal
	// See http://algorithms.tutorialhorizon.com/binary-tree-preorder-traversal-non-recursive-approach/ for more info.

	// 1. Create a Stack
	vector<ER *> tree_stack;
	ER *root = er;
	root->parent = root;
	int n = 0;

save_step_2:
	// 2. Print the root and push it to Stack and go left, i.e root=root.left and till it hits the nullptr.
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
	}
	

	// 3. If root is null and Stack is empty Then
	//		return, we are done.
	if (root == nullptr && tree_stack.empty())
	{
		return;
	}

	// 4. Else
	//		Pop the top Node from the Stack and set it as, root = popped_Node.
	//		Go right, root = root.right.
	//		Go to step 2.

	root = tree_stack.back();
	tree_stack.pop_back();
	
	if (!root->done)
	{
		ERs overlapped;
		ER *parent = root;
		while ((root->bound&parent->bound).area() / (double)parent->bound.area() > OVERLAP_COEF && (!parent->done))
		{
			parent->done = true;
			overlapped.push_back(parent);
			parent = parent->parent;
		}
		
		// Core part of NMS
		// Rt-k is the parent of Rt in component tree
		// Remove ERs such that number of overlap < 3, select the one with highest stability
		// If there exist 2 or more overlapping ER with same stability, choose the one having smallest area
		// overlap		O(Rt-k, Rt) = |Rt| / |Rt-k|
		// stability	S(Rt) = (|Rt-t'| - |Rt|) / |Rt|
		if (overlapped.size() >= 1 + STABILITY_T)
		{
			for (int i = 0; i < overlapped.size() - STABILITY_T; i++)
			{
				overlapped[i]->stability = (overlapped[i + STABILITY_T]->bound.area() - overlapped[i]->bound.area()) / (double)overlapped[i]->bound.area();
			}

			int min = 0;
			for (int i = 1; i < overlapped.size() - STABILITY_T; i++)
			{
				if (overlapped[i]->stability < overlapped[min]->stability )
					min = i;
				else if (overlapped[i]->stability == overlapped[min]->stability)
					min = (overlapped[i]->bound.area() < overlapped[min]->bound.area()) ? i : min;
			}

			double aspect_ratio = (double)overlapped[min]->bound.width / (double)overlapped[min]->bound.height;
			if (aspect_ratio < 2.5 && aspect_ratio > 0.1 && 
				overlapped[min]->area < MAX_AREA && 
				overlapped[min]->bound.height < input.rows*0.7 &&
				overlapped[min]->bound.width < input.cols*0.7)
			{
				pool.push_back(overlapped[min]);
				char buf[20];
				sprintf(buf, "res/tmp/%d.jpg", n++);
				imwrite(buf, input(overlapped[min]->bound));
			}
		}
	}

	root = root->next;
	goto save_step_2;

	// 5. End If
}

void ERFilter::classify(ERs pool, Mat input)
{
	int k = 0;
	const int N = 2;
	const int normalize_size = 24;
	
	for (auto it : pool)
	{
		vector<double> spacial_hist = make_LBP_hist(input(it->bound), 2, 24);
		rectangle(input, it->bound, Scalar(255));
		//cout << k << " " << adb->predict(spacial_hist) << endl;
		if (adb->predict(spacial_hist) > 0)
		{
			
			/*double ocr_result = ocr->chain_run(input(it->bound), 0);
			it->c = floor(ocr_result);
			it->prob = ocr_result - it->c;
			char buf[20];
			sprintf(buf, "res/tmp2/%d.jpg", k);
			imwrite(buf, input(it->bound));
			cout << k << " ";*/
		}
		k++;
	}
}

vector<double> ERFilter::make_LBP_hist(Mat input, const int N, const int normalize_size)
{
	const int block_size = normalize_size / N;
	const int bins = 256;


	Mat LBP = calc_LBP(input, normalize_size);
	vector<double> spacial_hist(N * N * bins);

	// for each sub-region
	for (int m = 0; m < N; m++)
	{
		for (int n = 0; n < N; n++)
		{
			// for each pixel in sub-region
			for (int i = 0; i < block_size; i++)
			{
				uchar* ptr = LBP.ptr(m*block_size + i, n*block_size);
				for (int j = 0; j < block_size; j++)
				{
					spacial_hist[m*N * bins + n * bins + ptr[j]]++;
				}
			}
		}
	}

	return spacial_hist;
}

Mat ERFilter::calc_LBP(Mat input, const int size)
{
	ocr->ARAN(input, input, size);
	//resize(input, input, Size(size, size));


	Mat LBP = Mat::zeros(size, size, CV_8U);
	for (int i = 1; i < size-1; i++)
	{
		uchar* ptr_input = input.ptr<uchar>(i);
		uchar* ptr = LBP.ptr<uchar>(i);
		for (int j = 1; j < size-1; j++)
		{
			double thresh = (ptr_input[j - size - 1] + ptr_input[j - size] + ptr_input[j - size + 1] + ptr_input[j + 1] +
				ptr_input[j + size + 1] + ptr_input[j + size] + ptr_input[j + size - 1] + ptr_input[j - 1]) / 9.0;

			ptr[j] += (ptr_input[j - size - 1] > thresh) << 0;
			ptr[j] += (ptr_input[j - size] > thresh) << 1;
			ptr[j] += (ptr_input[j - size + 1] > thresh) << 2;
			ptr[j] += (ptr_input[j + 1] > thresh) << 3;
			ptr[j] += (ptr_input[j + size + 1] > thresh) << 4;
			ptr[j] += (ptr_input[j + size] > thresh) << 5;
			ptr[j] += (ptr_input[j + size - 1] > thresh) << 6;
			ptr[j] += (ptr_input[j - 1] > thresh) << 7;

			/*ptr[j] += (ptr_input[j - size] > ptr_input[j]) << 0;
			ptr[j] += (ptr_input[j + 1] > ptr_input[j]) << 1;
			ptr[j] += (ptr_input[j + size - 1] > ptr_input[j]) << 2;*/
		}
	}
	return LBP;
}

