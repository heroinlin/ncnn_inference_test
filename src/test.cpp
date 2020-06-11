#include "stdlib.h"
#include "stdio.h"
#include "vector"
#include "string"
#include "time_utility.h"
#include <numeric>
#include <complex>
#include "ncnn/net.h"
#include "npy.hpp"
using namespace std;

ncnn::Net ncnn_net;
//ncnn_net.use_int8_inference = 0;
//ncnn_net.use_winograd_convolution = 0;


void load_model(char *model_path, char *param_path)
{
	int res = 0;
	res = ncnn_net.load_param(param_path);
	if (res != 0)
	{
		printf("load param failed!\n");
		return;
	}
	res = ncnn_net.load_model(model_path);
	if (res != 0)
	{
		printf("load model failed!\n");
		return;
	}
	printf("load model finish!\n");
}


void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}


void ncnn_test_npy(char *model_path, char *param_path, char *input_name, char *output_name, std::string image_path, char* output_save_path, int print_flag)
{
	/*const float mean_vals[3] = { 0.4914, 0.4822, 0.4465 };
	const float norm_vals[3] = { 0.247, 0.243, 0.261 };*/
	int res = 0;
	reid_load_model(model_path, param_path);
	std::vector<int> shape;
	std::vector<float> data;

	std::cout << "start read" << std::endl;

	aoba::LoadArrayFromNumpy(image_path, shape, data);
	std::vector<float> image(data.begin(), data.end());
	unsigned long shape_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned long>());

	std::cout << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
	ncnn::Mat in(shape[2], shape[1], shape[0], image.data(), 4);/*w, h, c*/

	// ncnn::Mat in(224, 224, 3, image.data(), 4);
	//for (int i = 0; i <3; i++)
	//	for (int j = 0; j < 20; j++)
	//	{
	//		printf("%d=%0.7f\n", i, ((float*)in.channel(i))[j]);
	//	}
	//in.substract_mean_normalize(mean_vals, norm_vals);
	
	ncnn::Extractor ex = ncnn_net.create_extractor();
	ex.set_light_mode(false);

	res = ex.input(input_name, in);
	ncnn::Mat out;

	std::cout << "start inference..." << std::endl;
	evaluate_time(res = ex.extract(output_name, out);, "compute time: ");
	std::cout << "out dim: " << out.dims <<", "<< out.c << "x" << out.h << "x" << out.w << std::endl;
	/*FILE *f = fopen(output_save_path, "wb+");
	for (int i = 0; i < out.c; i++)
	{
		int fnums = out.w*out.h;
		unsigned char * p = (unsigned char*)out.data + out.cstep * i * out.elemsize;
		fwrite(p, fnums*out.elemsize, 1, f);
	}
	
	fclose(f);*/
        if (print_flag){
	    pretty_print(out);
	}
	
}


int main(int argc, char *argv[])
{
	if (argc <5)
	{
		std::cout << "Usage: ncnn_test.exe <model_path> <param_path> <input_name> <output_name> <input_path>" << std::endl;
		return -1;
	}

	char *model_path = argv[1];
	char *param_path = argv[2];
    char *input_name = argv[3];
    char *output_name = argv[4];
	char *image_path = argv[5];
	char *output_save_path = argv[6];
	int print_flag = 1;
	if (argc >=7)
	{
		print_flag = atoi(argv[7]);
 	}
	ncnn_test_npy(model_path, param_path,input_name, output_name,  image_path, output_save_path, print_flag);
	return 1;
}

