#include "../inc/ER.h"

// ====================================================
// ======================== ER ========================
// ====================================================
ER::ER(const int level_, const int pixel_, const int x_, const int y_) : level(level_), pixel(pixel_), x(x_), y(y_), area(1), done(false), stability(.0), 
																		parent(nullptr), child(nullptr), next(nullptr), sibling_L(nullptr), sibling_R(nullptr), stkw(0)
{
	bound = Rect(x_, y_, 1, 1);
}

// ====================================================
// ===================== ER_filter ====================
// ====================================================
ERFilter::ERFilter(int thresh_step, int min_area, int max_area, int stability_t, double overlap_coef, double min_ocr_prob) : THRESH_STEP(thresh_step), MIN_AREA(min_area), MAX_AREA(max_area),
																													STABILITY_T(stability_t), OVERLAP_COEF(overlap_coef), MIN_OCR_PROB(min_ocr_prob)
{

}

void ERFilter::set_thresh_step(int t)
{
	THRESH_STEP = t;
}


void ERFilter::set_min_area(int m)
{
	MIN_AREA = m;
}


vector<double> ERFilter::text_detect(Mat src, ERs &root, vector<ERs> &all, vector<ERs> &pool, vector<ERs> &strong, vector<ERs> &weak, ERs &tracked, vector<Text> &text)
{
	chrono::high_resolution_clock::time_point start, end;
	start = chrono::high_resolution_clock::now();

	Mat Ycrcb;
	vector<Mat> channel;
	compute_channels(src, Ycrcb, channel);

	root.resize(channel.size());
	all.resize(channel.size());
	pool.resize(channel.size());
	strong.resize(channel.size());
	weak.resize(channel.size());

	vector<chrono::high_resolution_clock::time_point> time_vec(channel.size() * 4 + 4);

#pragma omp parallel for
	for (int i = 0; i < channel.size(); i++)
	{
		time_vec[i*4] = chrono::high_resolution_clock::now();
		root[i] = er_tree_extract(channel[i]);
		time_vec[i*4+1] = chrono::high_resolution_clock::now();
		non_maximum_supression(root[i], all[i], pool[i], channel[i]);
		time_vec[i*4+2] = chrono::high_resolution_clock::now();
		classify(pool[i], strong[i], weak[i], channel[i]);
		time_vec[i*4+3] = chrono::high_resolution_clock::now();
	}

	time_vec.rbegin()[3] = chrono::high_resolution_clock::now();
	er_track(strong, weak, tracked, channel, Ycrcb);
	time_vec.rbegin()[2] = chrono::high_resolution_clock::now();
#ifndef DO_OCR
	er_grouping(tracked, text, false, false);
	time_vec.rbegin()[1] = chrono::high_resolution_clock::now();
#else
	er_grouping(tracked, text, false, true);
	time_vec.rbegin()[1] = chrono::high_resolution_clock::now();
	er_ocr(tracked, channel, text);
	time_vec.rbegin()[0] = chrono::high_resolution_clock::now();
#endif

	end = chrono::high_resolution_clock::now();


	// calculate time
	chrono::duration<double> extract_time;
	chrono::duration<double> nms_time;
	chrono::duration<double> classify_time;
	for (int i = 0; i < channel.size(); i++)
	{
		if(extract_time < time_vec[i * 4 + 1] - time_vec[i * 4])
			extract_time = time_vec[i * 4 + 1] - time_vec[i * 4];

		if(nms_time < time_vec[i * 4 + 2] - time_vec[i * 4 + 1])
			nms_time = time_vec[i * 4 + 2] - time_vec[i * 4 + 1];

		if(classify_time < time_vec[i * 4 + 3] - time_vec[i * 4 + 2])
			classify_time = time_vec[i * 4 + 3] - time_vec[i * 4 + 2];
	}
	chrono::duration<double> track_time = (time_vec.rbegin()[2] - time_vec.rbegin()[3]);
	chrono::duration<double> grouping_time = (time_vec.rbegin()[1] - time_vec.rbegin()[2]);
#ifdef DO_OCR
	chrono::duration<double> ocr_time = (time_vec.rbegin()[0] - time_vec.rbegin()[1]);
#endif

	vector<double> times(7, 0);
	times[0] = extract_time.count();
	times[1] = nms_time.count();
	times[2] = classify_time.count();
	times[3] = track_time.count();
	times[4] = grouping_time.count();
#ifdef DO_OCR
	times[5] = ocr_time.count();
#endif
	times[6] = chrono::duration<double>(end - start).count();

	return times;
}


void ERFilter::compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels)
{
	vector<Mat> splited;
	channels.clear();

	cv::cvtColor(src, YCrcb, COLOR_BGR2YCrCb);
	split(YCrcb, splited);

	channels.push_back(splited[0]);
	channels.push_back(splited[1]);
	channels.push_back(splited[2]);
	channels.push_back(255 - splited[0]);
	channels.push_back(255 - splited[1]);
	channels.push_back(255 - splited[2]);
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

	/*plist *ptr = er->p;
	while (ptr != nullptr)
	{
		ptr = ptr->next;
	}
	ptr = new plist(current_pixel);*/
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

	if (child->area <= MIN_AREA)
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

	/*child->next = parent->child;
	parent->child = child;
	child->parent = parent;*/
}


void ERFilter::er_delete(ER *er)
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

	ER *to_delete = root;
	root = root->next;
	delete to_delete;
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

		//!< 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the
		//!<	returned pixel is at the same grey - level as the previous, go to 4	
		if (priority == highest_level)
		{
			delete[] pixel_accessible;
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


void ERFilter::process_stack(const int new_pixel_grey_level, ERs &er_stack)
{
	do
	{
		//!< 1. Process component on the top of the stack. The next grey-level is the minimum
		//!<	of new_pixel_grey_level and the grey - level for the second component on
		//!<	the stack.
		ER *top = er_stack.back();
		ER *second_top = er_stack.end()[-2];
		er_stack.pop_back();

		//!< 2. If new_pixel_grey_level is smaller than the grey-level on the second component
		//!<	on the stack, set the top of stack grey-level to new_pixel_grey_level and return
		//!<	from sub - routine(This occurs when the new pixel is at a grey-level for which
		//!<	there is not yet a component instantiated, so we let the top of stack be that
		//!<	level by just changing its grey - level.
		if (new_pixel_grey_level < second_top->level)
		{
			er_stack.push_back(new ER(new_pixel_grey_level, top->pixel, top->x, top->y));
			er_merge(er_stack.back(), top);
			return;
		}

		//!< 3. Remove the top of stack and merge it into the second component on stack
		//!<	as follows : Add the first and second moment accumulators together and / or
		//!<	join the pixel lists.Either merge the histories of the components, or take the
		//!<	history from the winner.Note here that the top of stack should be considered
		//!<	one ’time-step’ back, so its current size is part of the history.Therefore the
		//!<	top of stack would be the winner if its current size is larger than the previous
		//!<	size of second on stack.
		//er_stack.pop_back();
		er_merge(second_top, top);
		
	}
	//!< 4. If(new_pixel_grey_level>top_of_stack_grey_level) go to 1.
	while (new_pixel_grey_level > er_stack.back()->level);
}


void ERFilter::non_maximum_supression(ER *er, ERs &all, ERs &pool, Mat input)
{
	// Non Recursive Preorder Tree Traversal
	// See http://algorithms.tutorialhorizon.com/binary-tree-preorder-traversal-non-recursive-approach/ for more info.

	// 1. Create a Stack
	vector<ER *> tree_stack;
	ER *root = er;
	root->parent = root;

save_step_2:
	// 2. Print the root and push it to Stack and go left, i.e root=root.left and till it hits the nullptr.
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
	#ifdef GET_ALL_ER
		all.push_back(root);
	#endif
	}
	

	// 3. If root is null and Stack is empty Then
	//		return, we are done.
	if (root == nullptr && tree_stack.empty())
	{
		//cout << "Before NMS: " << n << "    After NMS: " << pool.size() << endl;
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
				overlapped[i]->stability =  (double)overlapped[i]->bound.area() / (double)(overlapped[i + STABILITY_T]->bound.area() - overlapped[i]->bound.area());
			}

			int max = 0;
			for (int i = 1; i < overlapped.size() - STABILITY_T; i++)
			{
				if (overlapped[i]->stability > overlapped[max]->stability )
					max = i;
				else if (overlapped[i]->stability == overlapped[max]->stability)
					max = (overlapped[i]->bound.area() < overlapped[max]->bound.area()) ? i : max;
			}

			double aspect_ratio = (double)overlapped[max]->bound.width / (double)overlapped[max]->bound.height;
			if (aspect_ratio < 2.0 && aspect_ratio > 0.10 && 
				overlapped[max]->area < MAX_AREA && 
				overlapped[max]->area > MIN_AREA &&
				overlapped[max]->bound.height < input.rows*0.8 &&
				overlapped[max]->bound.width < input.cols*0.8)
			{
				pool.push_back(overlapped[max]);
				/*char buf[20];
				sprintf(buf, "res/tmp/%d.jpg", n++);
				imwrite(buf, input(overlapped[max]->bound));*/
			}
		}
	}

	root = root->next;
	goto save_step_2;

	// 5. End If
}

void ERFilter::classify(ERs &pool, ERs &strong, ERs &weak, Mat input)
{
	int k = 0;
	const int N = 2;
	const int normalize_size = 24;

	for (int i = 0; i < pool.size(); i++)
	{
		vector<double> fv = make_LBP_hist(input(pool[i]->bound), N, normalize_size);
		if (stc->predict(fv) > -DBL_MAX)
		{
			strong.push_back(pool[i]);
		}
		else
		{
			if (wtc->predict(fv) > -DBL_MAX)
			{
				weak.push_back(pool[i]);
			}
		}
	}
}



void ERFilter::er_track(vector<ERs> &strong, vector<ERs> &weak, ERs &all_er, vector<Mat> &channel, Mat Ycrcb)
{
#ifdef USE_STROKE_WIDTH
	StrokeWidth SWT;
#endif
	for (int i = 0; i < strong.size(); i++)
	{
		for (auto it : strong[i])
		{
			calc_color(it, channel[i], Ycrcb);
#ifdef USE_STROKE_WIDTH
			it->stkw = SWT.SWT(channel[i](it->bound));
#endif
			it->center = Point(it->bound.x + it->bound.width / 2, it->bound.y + it->bound.height / 2);
			it->ch = i;
		}

		for (auto it : weak[i])
		{
			calc_color(it, channel[i], Ycrcb);
#ifdef USE_STROKE_WIDTH
			it->stkw = SWT.SWT(channel[i](it->bound));
#endif
			it->center = Point(it->bound.x + it->bound.width / 2, it->bound.y + it->bound.height / 2);
			it->ch = i;
		}
	}

	for (int i = 0; i < strong.size(); i++)
	{
		all_er.insert(all_er.end(), strong[i].begin(), strong[i].end());
	}

	vector<vector<bool>> tracked(weak.size());
	for (int i = 0; i < tracked.size(); i++)
	{
		tracked[i].resize(weak[i].size());
	}


	for (int i = 0; i < all_er.size(); i++)
	{
		ER *s = all_er[i];
		for (int m = 0; m < weak.size(); m++)
		{
			for (int n = 0; n < weak[m].size(); n++)
			{
				if (tracked[m][n] == true) continue;

				ER* w = weak[m][n];
				if (abs(s->center.x - w->center.x) + abs(s->center.y - w->center.y) < max(s->bound.width, s->bound.height) << 1 &&
					abs(s->bound.height - w->bound.height) < min(s->bound.height, w->bound.height) &&
					abs(s->bound.width - w->bound.width) < (s->bound.width + w->bound.width) >> 1 &&
					abs(s->color1 - w->color1) < 25 &&
					abs(s->color2 - w->color2) < 25 &&
					abs(s->color3 - w->color3) < 25 &&
#ifdef USE_STROKE_WIDTH
					(s->stkw / w->stkw) < 4 &&
					(s->stkw / w->stkw) > 0.25 &&
#endif
					abs(s->area - w->area) < min(s->area, w->area) * 3)
				{
					tracked[m][n] = true;
					all_er.push_back(w);
				}
			}
		}
	}

	/*sort(all_er.begin(), all_er.end(), [](ER *a, ER *b) { return a->center.x < b->center.x; });
	for (int i = 0; i < all_er.size(); i++)
	{
		char buf[20];
		sprintf(buf, "res/tmp1/%d.jpg", i);
		imwrite(buf, Ycrcb(all_er[i]->bound));
		cout << i << " ";
	}*/
}


void ERFilter::er_grouping(ERs &all_er, vector<Text> &text, bool overlap_sup, bool inner_sup)
{
	sort(all_er.begin(), all_er.end(), [](ER *a, ER *b) { return a->center.x < b->center.x; });
	
	if(overlap_sup)
		overlap_suppression(all_er);
	if(inner_sup)
		inner_suppression(all_er);
	
	vector<int> group_index(all_er.size(), -1);
	int index = 0;
	for (int i = 0; i < all_er.size(); i++)
	{
		ER *a = all_er[i];
		for (int j = i+1; j < all_er.size(); j++)
		{
			ER *b = all_er[j];
			if (abs(a->center.x - b->center.x) < max(a->bound.width,b->bound.width)*3.0 &&
				abs(a->center.y - b->center.y) < (a->bound.height + b->bound.height)*0.25 &&			// 0.5*0.5
				abs(a->bound.height - b->bound.height) < min(a->bound.height, b->bound.height) &&
				abs(a->bound.width - b->bound.width) < min(a->bound.height, b->bound.height * 2) &&
				abs(a->color1 - b->color1) < 25 &&
				abs(a->color2 - b->color2) < 25 &&
				abs(a->color3 - b->color3) < 25 &&
#ifdef USE_STROKE_WIDTH
				(a->stkw / b->stkw) < 4 &&
				(a->stkw / b->stkw) > 0.25 &&
#endif
				abs(a->area - b->area) < min(a->area, b->area)*4)
			{
				if (group_index[i] == -1 && group_index[j] == -1)
				{
					group_index[i] = index;
					group_index[j] = index;
					text.push_back(Text());
					text[index].ers.push_back(a);
					text[index].ers.push_back(b);
					index++;
				}

				else if (group_index[j] != -1)
				{
					group_index[i] = group_index[j];
					text[group_index[i]].ers.push_back(a);
				}

				else
				{
					group_index[j] = group_index[i];
					text[group_index[j]].ers.push_back(b);
				}
			}
		}
	}

	for (int i = 0; i < text.size(); i++)
	{
		sort(text[i].ers.begin(), text[i].ers.end(), [](ER *a, ER *b) { return a->center.x < b->center.x; });

		ERs tmp_ers;
		tmp_ers.assign(text[i].ers.begin(), text[i].ers.end());
		overlap_suppression(tmp_ers);
		inner_suppression(tmp_ers);

		vector<Point> points;
		for (int j = 0; j < tmp_ers.size(); j++)
		{
			points.push_back(tmp_ers[j]->bound.br());
		}

		text[i].slope = fitline_avgslope(points);
		//cout << text[i].slope << endl;
#ifndef DO_OCR
		text[i].box = text[i].ers.front()->bound;
		for (int j = 0; j < text[i].ers.size(); j++)
		{
			text[i].box |= text[i].ers[j]->bound;
		}
#endif	
	}
}


void ERFilter::er_ocr(ERs &all_er, vector<Mat> &channel, vector<Text> &text)
{
	const unsigned min_er = 6;
	const unsigned min_pass_ocr = 2;
	
	for (int i = text.size()-1; i >= 0; i--)
	{
		// delete ERs that are in the same channel and are highly overlap
		vector<bool> to_delete(text[i].ers.size(), false);
		for (int m = 0; m < text[i].ers.size(); m++)
		{
			for (int n = m + 1; n < text[i].ers.size(); n++)
			{
				double overlap_area = (text[i].ers[m]->bound & text[i].ers[n]->bound).area();
				double union_area = (text[i].ers[m]->bound | text[i].ers[n]->bound).area();
				if (overlap_area / union_area > 0.95)
				{
					if (text[i].ers[m]->bound.area() > text[i].ers[n]->bound.area())
						to_delete[n] = true;
					else
						to_delete[m] = true;
				}
			}
		}

		for (int j = text[i].ers.size() - 1; j >= 0; j--)
		{
			if (to_delete[j])
				text[i].ers.erase(text[i].ers.begin() + j);
		}


		// get OCR label of each ER
	#pragma omp parallel for
		for (int j = 0; j < text[i].ers.size(); j++)
		{
			ER* er = text[i].ers[j];
			const double result = ocr->chain_run(channel[er->ch](er->bound), er->level*THRESH_STEP, text[i].slope);
			er->letter = floor(result);
			er->prob = result - floor(result);
		}
		
		// delete ER with low OCR confidence
		for (int j = text[i].ers.size() - 1; j >= 0; j--)
		{
			if (text[i].ers[j]->prob < MIN_OCR_PROB)
				text[i].ers.erase(text[i].ers.begin() + j);
		}
		
		if (text[i].ers.size() < min_pass_ocr)
		{
			text.erase(text.begin() + i);
			continue;
		}

		Graph graph;
		build_graph(text[i], graph);
		solve_graph(text[i], graph);
		ocr->feedback_verify(text[i]);

		/*fstream fout("graph.txt", fstream::out);
		for (int i = 0; i < graph.size(); i++)
		{
			fout << graph[i].index << " " << graph[i].vertex->letter << " " << graph[i].vertex->prob * 100 << " ";
			for (int j = 0; j < graph[i].adj_list.size(); j++)
			{
				fout << graph[i].adj_list[j].index << " ";
				fout << graph[i].edge_prob[j] * 50 << " ";
			}
			fout << endl;

			char buf[30];
			sprintf(buf, "D:/%d.PNG", i);
			Mat ocr_img = channel[graph[i].vertex->ch](graph[i].vertex->bound);
			double resize_factor = 30.0 / ocr_img.rows;
			resize(ocr_img, ocr_img, Size(), resize_factor, resize_factor);
			threshold(ocr_img, ocr_img, 128, 255, THRESH_OTSU);
			imwrite(buf, ocr_img);
		}
		fout.close();*/

		
		text[i].box = text[i].ers.front()->bound;
		for (int j = 0; j < text[i].ers.size(); j++)
		{
			text[i].box |= text[i].ers[j]->bound;
		}

		spell_check(text[i]);
		//cout << text[i].word << " " << text[i].slope << endl;
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
	ocr->ARAN(input, input, size + 2);
	//resize(input, input, Size(size + 2, size + 2));

	Mat LBP = Mat::zeros(size, size, CV_8U);
	for (int i = 0; i < size; i++)
	{
		uchar* ptr_input = input.ptr<uchar>(i + 1, 1);
		uchar* ptr = LBP.ptr<uchar>(i);
		for (int j = 0; j < size; j++)
		{
			double thresh = (ptr_input[j - size - 1] + ptr_input[j - size] + ptr_input[j - size + 1] + ptr_input[j + 1] +
				ptr_input[j + size + 1] + ptr_input[j + size] + ptr_input[j + size - 1] + ptr_input[j - 1]) / 8.0;

			ptr[j] += (ptr_input[j - size - 1] > thresh) << 0;
			ptr[j] += (ptr_input[j - size] > thresh) << 1;
			ptr[j] += (ptr_input[j - size + 1] > thresh) << 2;
			ptr[j] += (ptr_input[j + 1] > thresh) << 3;
			ptr[j] += (ptr_input[j + size + 1] > thresh) << 4;
			ptr[j] += (ptr_input[j + size] > thresh) << 5;
			ptr[j] += (ptr_input[j + size - 1] > thresh) << 6;
			ptr[j] += (ptr_input[j - 1] > thresh) << 7;
		}
	}
	return LBP;
}



inline bool ERFilter::is_neighboring(ER *a, ER *b)
{
	const double T1 = 1.8;		// height ratio
	const double T2 = 3.0;		// x distance
	const double T3 = 0.5;		// y distance
	const double T4 = 3.0;		// area ratio
	const double T5 = 25;		// color

	double height_ratio = max(a->bound.height, b->bound.height) / (double)min(a->bound.height, b->bound.height);
	double x_d = abs(a->center.x - b->center.x);
	double y_d = abs(a->center.y - b->center.y);
	double area_ratio = max(a->area, b->area) / (double)min(a->area, b->area);
	double color1 = abs(a->color1 - b->color1);
	double color2 = abs(a->color2 - b->color2);
	double color3 = abs(a->color3 - b->color3);

	if ((1 / T1 < height_ratio) && (height_ratio < T1) &&
		x_d < T2 * max(a->bound.width, b->bound.width) &&
		y_d < T3 * min(a->bound.height, b->bound.height) &&
		(1 / T4 < area_ratio) && (area_ratio < T4) &&
		color1 < T5 && color2 < T5 && color3 < T5)
		return true;
		

	else
		return false;
}

inline bool ERFilter::is_overlapping(ER *a, ER *b)
{
	const double T1 = 0.7;	// area overlap
	const double T2 = 0.5;	// distance

	Rect intersect = a->bound & b->bound;

	if (intersect.area() > T1 * min(a->bound.area(), b->bound.area()) &&
		norm(a->center - b->center) < T2 * max(a->bound.height, b->bound.height))
		return true;

	else
		return false;
}


void ERFilter::inner_suppression(ERs &pool)
{
	vector<bool> to_delete(pool.size(), false);
	const double T1 = 2.0;
	const double T2 = 0.2;

	for (int i = 0; i < pool.size(); i++)
	{
		for (int j = 0; j < pool.size(); j++)
		{
			if (norm(pool[i]->center - pool[j]->center) < T2 * max(pool[i]->bound.width, pool[i]->bound.height))
			{
				if (pool[i]->bound.x <= pool[j]->bound.x &&
					pool[i]->bound.y <= pool[j]->bound.y &&
					pool[i]->bound.br().x >= pool[j]->bound.br().x &&
					pool[i]->bound.br().y >= pool[j]->bound.br().y &&
					(double)pool[i]->bound.area() / (double)pool[j]->bound.area() > T1)
					to_delete[j] = true;
			}
		}
	}



	for (int i = pool.size() - 1; i >= 0; i--)
	{
		if (to_delete[i])
			pool.erase(pool.begin() + i);
	}
}


void ERFilter::overlap_suppression(ERs &pool)
{
	vector<bool> merged(pool.size(), false);

	for (int i = 0; i < pool.size(); i++)
	{
		for (int j = i + 1; j < pool.size(); j++)
		{
			if (merged[j])	continue;

			Rect overlap = pool[i]->bound & pool[j]->bound;
			Rect union_box = pool[i]->bound | pool[j]->bound;
			
			if ((double)overlap.area() / (double)union_box.area() > 0.5)
			{
				merged[j] = true;

				int x = (pool[i]->bound.x + pool[j]->bound.x) * 0.5;
				int y = (pool[i]->bound.y + pool[j]->bound.y) * 0.5;
				int width = (pool[i]->bound.width + pool[j]->bound.width) * 0.5;
				int height = (pool[i]->bound.height + pool[j]->bound.height) * 0.5;

				pool[i]->bound.x = x;
				pool[i]->bound.y = y;
				pool[i]->bound.height = height;
				pool[i]->bound.width = width;			
				pool[i]->center.x = x + pool[i]->bound.width * 0.5;
				pool[i]->center.y = y + pool[i]->bound.height * 0.5;
			}
		}
	}

	for (int i = pool.size()-1; i >= 0; i--)
	{
		if (merged[i])
		{
			pool.erase(pool.begin() + i);
		}
	}
}


// model as a graph problem
void ERFilter::build_graph(Text &text, Graph &graph)
{
	for (int j = 0; j < text.ers.size(); j++)
		graph.push_back(GraphNode(text.ers[j], j));

	for (int j = 0; j < text.ers.size(); j++)
	{
		bool found_next = false;
		int cmp_idx = -1;
		for (int k = j + 1; k < text.ers.size(); k++)
		{
			// encounter an ER that is overlapping to j
			if (is_overlapping(text.ers[j], text.ers[k]))
				continue;

			// encounter an ER that is the first one different from j
			else if (!found_next)
			{
				found_next = true;
				cmp_idx = k;

				const int a = ocr->index_mapping(graph[j].vertex->letter);
				const int b = ocr->index_mapping(graph[k].vertex->letter);
				graph[j].edge_prob.push_back(tp[a][b]);
				graph[j].adj_list.push_back(graph[k]);
			}

			// encounter an ER that is overlapping to cmp_idx
			else if (is_overlapping(text.ers[cmp_idx], text.ers[k]))
			{
				cmp_idx = k;

				const int a = ocr->index_mapping(graph[j].vertex->letter);
				const int b = ocr->index_mapping(graph[k].vertex->letter);
				graph[j].edge_prob.push_back(tp[a][b]);
				graph[j].adj_list.push_back(graph[k]);
			}

			// encounter an ER that is different from cmp_idx, the stream is ended
			else
				break;
		}
	}
}


// solve the graph problem by Dynamic Programming
void ERFilter::solve_graph(Text &text, Graph &graph)
{
	vector<double> DP_score(graph.size(), 0);
	vector<int> DP_path(graph.size(), -1);
	const double char_weight = 100;
	const double edge_weight = 50;

	for (int j = 0; j < graph.size(); j++)
	{
		if (DP_path[j] == -1)
			DP_score[j] = graph[j].vertex->prob * char_weight;

		for (int k = 0; k < graph[j].adj_list.size(); k++)
		{
			const int &adj = graph[j].adj_list[k].index;
			const double score = DP_score[j] + graph[j].edge_prob[k] * edge_weight + text.ers[adj]->prob * char_weight;
			
			if (score > DP_score[adj])
			{
				DP_score[adj] = score;
				DP_path[adj] = j;
			}
		}
	}

	// construct the optimal path
	double max = 0;
	int arg_max;
	for (int j = 0; j < DP_score.size(); j++)
	{
		if (DP_score[j] > max)
		{
			max = DP_score[j];
			arg_max = j;
		}
	}

	int node_idx = arg_max;

	text.ers.clear();
	while (node_idx != -1)
	{
		text.ers.push_back(graph[node_idx].vertex);
		node_idx = DP_path[node_idx];
	}

	reverse(text.ers.begin(), text.ers.end());

	for (auto it : text.ers)
		text.word.append(string(1, it->letter));
}


void ERFilter::spell_check(Text &text)
{
	if (text.word.length() <= 1)
		return;

	for (auto it : text.word)
	{
		if (it >= '0' && it <= '9')	return;
	}

	string request = text.word;
	transform(request.begin(), request.end(), request.begin(), ::tolower);

	//cout << text.word << " -> ";
	string corrected = corrector.correct(request);
	int upper_count = 0;
	int lower_count = 0;
	for (int i = 0; i < corrected.length(); i++)
	{
		if (i < text.word.length())
		{
			if (text.word[i] >= 'A' && text.word[i] <= 'Z')
			{
				text.word[i] = corrected[i] - 0x20;
				++upper_count;
			}
				
			else
			{
				text.word[i] = corrected[i];
				++lower_count;
			}
		}
		
		else
		{
			if (upper_count > lower_count)
			{
				text.word.push_back(corrected[i] - 0x20);
				++upper_count;
			}
			else
			{
				text.word.push_back(corrected[i]);
				++upper_count;
			}
				
			text.box.width *= (1.0 + 1.0 / text.word.size());
		}
	}
	

	if ((double)upper_count/(upper_count + lower_count) > 0.6)
	{
		for (int i = 0; i < text.word.length(); i++)
		{
			if (text.word[i] >= 'a' && text.word[i] <= 'z')
			{
				text.word[i] -= 0x20;
			}
		}
	}
	//cout << corrected << endl;
}


bool ERFilter::load_tp_table(const char* filename)
{
	fstream fin;
	fin.open(filename, fstream::in);
	if (!fin.is_open())
	{
		std::cout << "Error: " << filename << " is not opened!!" << endl;
		return false;
	}

	string buffer;
	int i = 0;
	while (getline(fin, buffer))
	{
		int j = 0;
		istringstream row_string(buffer);
		string token;
		while (getline(row_string, token, ' '))
		{
			tp[i][j] = stof(token);
			j++;
		}
		i++;
	}

	return true;
}


double StrokeWidth::SWT(Mat input)
{
	Mat thresh;
	Mat canny;
	Mat blur;
	Mat grad_x;
	Mat grad_y;
	threshold(input, thresh, 128, 255, THRESH_OTSU);
	cv::Canny(thresh, canny, 150, 300, 3);
	cv::GaussianBlur(thresh, blur, Size(5, 5), 0);
	cv::Sobel(blur, grad_x, CV_32F, 1, 0, 3);
	cv::Sobel(blur, grad_y, CV_32F, 0, 1, 3);


	// Stroke Width Transform 1st pass
	Mat SWT_img(input.rows, input.cols, CV_32F, FLT_MAX);

	vector<Ray> rays;

	for (int i = 0; i < canny.rows; i++)
	{
		uchar *ptr = canny.ptr(i);
		float *xptr = grad_x.ptr<float>(i);
		float *yptr = grad_y.ptr<float>(i);
		for (int j = 0; j < canny.cols; j++)
		{
			if (ptr[j] != 0)
			{
				int x = j;
				int y = i;
				double dir_x = xptr[j] / sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
				double dir_y = yptr[j] / sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
				double cur_x = x;
				double cur_y = y;
				int cur_pixel_x = x;
				int cur_pixel_y = y;
				vector<SWTPoint2d> point;
				point.push_back(SWTPoint2d(x, y));
				for (;;)
				{
					cur_x += dir_x;
					cur_y += dir_y;

					if (round(cur_x) == cur_pixel_x && round(cur_y) == cur_pixel_y)
						continue;
					else
						cur_pixel_x = round(cur_x), cur_pixel_y = round(cur_y);

					if (cur_pixel_x < 0 || (cur_pixel_x >= canny.cols) || cur_pixel_y < 0 || (cur_pixel_y >= canny.rows))
						break;

					point.push_back(SWTPoint2d(cur_pixel_x, cur_pixel_y));
					double gx = grad_x.at<float>(cur_pixel_y, cur_pixel_x);
					double gy = grad_y.at<float>(cur_pixel_y, cur_pixel_x);
					double mag = sqrt(gx*gx + gy*gy);;
					double q_x = grad_x.at<float>(cur_pixel_y, cur_pixel_x) / mag;
					double q_y = grad_y.at<float>(cur_pixel_y, cur_pixel_x) / mag;
					if (acos(dir_x * -q_x + dir_y * -q_y) < CV_PI / 2.0)
					{
						double length = sqrt((cur_pixel_x - x)*(cur_pixel_x - x) + (cur_pixel_y - y)*(cur_pixel_y - y));
						for (auto it : point)
						{
							if (length < SWT_img.at<float>(it.y, it.x))
							{
								SWT_img.at<float>(it.y, it.x) = length;
							}
						}
						rays.push_back(Ray(SWTPoint2d(j, i), SWTPoint2d(cur_pixel_x, cur_pixel_y), point));
						break;
					}
				}
			}
		}
	}

	// Stroke Width Transform 2nd pass
	for (auto& rit : rays) {
		for (auto& pit : rit.points)
			pit.SWT = SWT_img.at<float>(pit.y, pit.x);

		std::sort(rit.points.begin(), rit.points.end(), [](SWTPoint2d lhs, SWTPoint2d rhs){return lhs.SWT < rhs.SWT; });
		float median = (rit.points[rit.points.size() / 2]).SWT;
		for (auto& pit : rit.points)
			SWT_img.at<float>(pit.y, pit.x) = std::min(pit.SWT, median);
	}


	// return mean stroke width
	double stkw = 0;
	int count = 0;
	for (int i = 0; i < SWT_img.rows; i++)
	{
		float* ptr = SWT_img.ptr<float>(i);
		for (int j = 0; j < SWT_img.cols; j++)
		{
			if (ptr[j] != FLT_MAX)
			{
				stkw += ptr[j];
				count++;
			}
		}
	}
	
	stkw /= count;
	return stkw;
}


inline void ColorHist::calc_hist(Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		uchar* ptr = img.ptr(i);
		for (int j = 0; j < img.cols * 3; j += 3)
		{
			c1[ptr[j]]++;
			c2[ptr[j + 1]]++;
			c3[ptr[j + 2]]++;
		}
	}

	const int total = img.total();
	for (int i = 0; i < 256; i++)
	{
		c1[i] /= total;
		c2[i] /= total;
		c3[i] /= total;
	}
}


inline double ColorHist::compare_hist(ColorHist ch)
{

}



double fitline_LSE(const vector<Point> &p)	// LSE works bad when there are both upper and lower line
{
	Mat A(p.size(), 2, CV_32F);
	Mat B(p.size(), 1, CV_32F);
	Mat AT;
	Mat invATA;

	for (int i = 0; i < p.size(); i++)
	{
		A.at<float>(i, 0) = p[i].x;
		A.at<float>(i, 1) = 1;
		B.at<float>(i) = p[i].y;
	}
	transpose(A, AT);
	invert(AT*A, invATA);

	Mat line = invATA*AT*B;

	return line.at<float>(0);
}


// fit the line uses with Least Medain of Square, the algorithm is described in 
// Peter J. Rousseeuw, "Least Median of Squares Regression", 1984, and
// J.M. Steele and W.L. Steiger, "Algorithms and complexity for Least Median of Squares regression", 1985
double fitline_LMS(const vector<Point> &p)
{
	// a line is express as y = alpha + beta * x
	double alpha_star = 0;
	double beta_star = 0;
	double d_star = DBL_MAX;

	for (int r = 0; r < p.size(); r++)
	{
		for (int s = r + 1; s < p.size(); s++)
		{
			double beta = (double)(p[r].y - p[s].y) / (double)(p[r].x - p[s].x);

			vector<double> z(p.size());
			for (int i = 0; i < p.size(); i++)
				z[i] = p[i].y - beta * p[i].x;

			sort(z.begin(), z.end(), [](double a, double b) {return a < b; });

			const int m = p.size() / 2;
			for (int j = 0; j < m; j++)
			{
				if (z[j + m] - z[j] < d_star)
				{
					d_star = z[j + m] - z[j];
					alpha_star = (z[j + m] + z[j]) / 2;
					beta_star = beta;
				}
			}
		}
	}

	// return the slope
	return beta_star;
}

double fitline_avgslope(const vector<Point> &p)
{
	if (p.size() <= 2)
		return 0;

	const double epsilon = 0.07;
	double slope = .0;

	for (int i = 0; i < p.size() - 2; i++)
	{
		double slope12 = (double)(p[i + 0].y - p[i + 1].y) / (p[i + 0].x - p[i + 1].x);
		double slope23 = (double)(p[i + 1].y - p[i + 2].y) / (p[i + 1].x - p[i + 2].x);
		double slope13 = (double)(p[i + 0].y - p[i + 2].y) / (p[i + 0].x - p[i + 2].x);

		if (abs(slope12 - slope23) < epsilon && abs(slope23 - slope13) < epsilon && abs(slope12 - slope13) < epsilon)
			slope += (slope12 + slope23 + slope13) / 3;
		else if (abs(slope12) < abs(slope23) && abs(slope12) < abs(slope13))
			slope += slope12;
		else if (abs(slope23) < abs(slope12) && abs(slope23) < abs(slope13))
			slope += slope23;
		else if (abs(slope13) < abs(slope12) && abs(slope13) < abs(slope23))
			slope += slope13;
	}
	
	slope /= (p.size() - 2);
	
	return slope;
}

void calc_color(ER* er, Mat mask_channel, Mat color_img)
{
	// calculate the color of each ER
	Mat img = mask_channel(er->bound).clone();
	threshold(255-img, img, 128, 255, THRESH_OTSU);

	int count = 0;
	double color1 = 0;
	double color2 = 0;
	double color3 = 0;
	for (int i = 0; i < img.rows; i++)
	{
		uchar* ptr = img.ptr(i);
		uchar* color_ptr = color_img.ptr(i);
		for (int j = 0, k = 0; j < img.cols; j++, k += 3)
		{
			if (ptr[j] != 0)
			{
				++count;
				color1 += color_ptr[k];
				color2 += color_ptr[k + 1];
				color3 += color_ptr[k + 2];
			}
		}
	}
	er->color1 = color1 / count;
	er->color2 = color2 / count;
	er->color3 = color3 / count;

	/*Mat img = color_img(er->bound);
	double color1 = 0;
	double color2 = 0;
	double color3 = 0;
	for (int i = 0; i < img.rows; i++)
	{
		uchar *ptr = img.ptr(i);
		for (int j = 0, k = 0; j < img.cols; j++, k+=3)
		{
			color1 += ptr[k];
			color2 += ptr[k + 1];
			color3 += ptr[k + 2];
		}
	}
	er->color1 = color1 / (img.rows*img.cols);
	er->color2 = color2 / (img.rows*img.cols);
	er->color3 = color3 / (img.rows*img.cols);*/
}


vector<vector<int> > comb(int N, int K)
{
	std::string bitmask(K, 1);	// K leading 1's
	bitmask.resize(N, 0);		// N-K trailing 0's

	vector<vector<int> > all_combination;
	int comb_counter = 0;
	// print integers and permute bitmask
	do {
		all_combination.push_back(vector<int>());
		for (int i = 0; i < N; ++i) // [0..N-1] integers
		{
			if (bitmask[i])
			{
				//std::cout << " " << i;
				all_combination[comb_counter].push_back(i);
			}
		}
		//std::cout << std::endl;
		comb_counter++;
	} while (std::prev_permutation(bitmask.begin(), bitmask.end()));

	return all_combination;
}

double standard_dev(vector<double> arr, bool normalize)
{
	const int N = arr.size();
	double avg = 0;

	for (auto it : arr)
		avg += it;
	avg /= N;

	double sum = 0;
	for (auto it : arr)
		sum += pow(it - avg, 2);

	double stdev = sqrt(sum / N);

	return (normalize) ? stdev / avg : stdev;
}