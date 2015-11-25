#include "wcvt_gpu.h"
#include <math.h>
#include<GL\freeglut.h>

static int width, height;

//to compute wcv, give differenct weights to the radius of cones.
void CVT::Display_vorCone(cv::Mat & img, int numOfsites)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPointSize(3);

	double max_radius=std::max(img.size().height, img.size().width);
	
	for (int i = 0; i<numOfsites; i++)
	{
		cells[i].coverage.clear();
		float x = cells[i].site.y;
		float y = img.size().height-0.5- cells[i].site.x;
		cv::Point pix((int)cells[i].site.x, (int)cells[i].site.y);
		float d= color2dist(img, pix);//검정일때 d=1. 
		

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(x, y, 0);
		
		//integer 연산이라 i=270일때 (0,1,14)가 나온다.
		//the brighter one should have larger radius.
		glColor3ub(i / (256 * 256), (i / 256)%256 , i%256 );
		glutSolidCone(max_radius*((2 - d)+0.5f*(1+d)), 1, 16, 1);

		//draw each sites.(i.e. center of a cone)
		//glColor3ub(0, 0, 0);
		//glBegin(GL_POINTS);
		//glVertex3d(0, 0, 1);
		//glEnd();
	}
	glFinish();//double buffering하면 안된다?

}


//buil the Weighted VOR once
void CVT::vor_GPU(cv::Mat &  img)
{
	Display_vorCone(img,cells.size());
	glReadPixels(0, 0, img.size().width, img.size().height, GL_RGBA, GL_UNSIGNED_BYTE, pixels); //이미지 points의 r,g,b를 읽어서 저장한다. 일렬로. 2차원배열을. row먼저 읽는다.

	for (int i = 0; i < img.size().width; i++)
	{
		for (int j = 0; j < img.size().height; j++)
		{
			int color = pixels[(i + img.size().width*j) * 4] * 256 * 256 + pixels[(i + img.size().width*j) * 4 + 1] * 256 + pixels[(i + img.size().width*j) * 4 + 2];
		
			cells[color].coverage.push_back(cv::Point(img.size().height - j - 1, i));//먼저 coverage를 비워야
		}
	}

	//remove empty cells...
	int size = cells.size();
	for (int i = 0; i <size; i++)
	{
		if (cells[i].coverage.empty())
		{
			cells[i] = cells.back();
			cells.pop_back();
			i--;
			size--;
		}
	}//end for i
}


void CVT::compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites)
{
	//inint 
	int site_size = sites.size();
	cells.resize(site_size);
	for (int i = 0; i < site_size; i++)
	{
		cells[i].site = sites[i];
	}

	float max_dist_moved = FLT_MAX;

	int width = img.size().width;
	int height = img.size().height;

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(180, 100);

	glutCreateWindow("CVT");
	glEnable(GL_DEPTH_TEST);

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, width, 0, height, -1, 0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	pixels = new unsigned char[4 * img.size().width*img.size().height]; //4를 곱하는 이유, 한 픽셀에 rgb,알파

	int iteration = 0;
	do
	{
		vor_GPU(img); //compute voronoi
		max_dist_moved = move_sites(img);
		if (debug) std::cout << "[" << iteration << "] max dist moved = " << max_dist_moved << std::endl;
		iteration++;
	} while (max_dist_moved>max_site_displacement && iteration < this->iteration_limit);
	delete[] pixels;
	
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//CVT construnctor에 들어가야

	////glutMainLoop에 들어가면 계속 그림을 그리느라 바빠서 stipple을 못그린다.
	//glutMainLoop();

	if (debug) cv::waitKey();
}
