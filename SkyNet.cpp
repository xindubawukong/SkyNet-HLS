#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,32, 0,0,0},    //data
{ "dwconv1e", 320,160,32, 320,160,32, 3,1,1},    //dwconv1
{ "pwconv1e", 320,160,32, 320,160,64, 1,1,0},    //pwconv1
{ "pool1e",   320,160,64, 160,80,64,  2,2,0},    //pool1
{ "dwconv2e", 160,80,64,  160,80,64,  3,1,1},    //dwconv2
{ "pwconv2e", 160,80,64,  160,80,96,  1,1,0},    //pwconv2
{ "pool2",   160,80,96,  80,40,96,   2,2,0},    //pool2
{ "dwconv3", 80,40,96,   80,40,96,   3,1,1},    //dwconv3
{ "pwconv3", 80,40,96,   80,40,192,  1,1,0},    //pwconv3
{ "reorg",   80,40,192,  40,20,768,  2,2,0},    //reorg
{ "pool3",   80,40,192,  40,20,192,  2,2,0},    //pool3
{ "dwconv4", 40,20,192,  40,20,192,  3,1,1},    //dwconv4
{ "pwconv4", 40,20,192,  40,20,384,  1,1,0},    //pwconv4
{ "dwconv5", 40,20,384,  40,20,384,  3,1,1},    //dwconv5
{ "pwconv5", 40,20,384,  40,20,512,  1,1,0},    //pwconv5
{ "cat",     40,20,192,  40,20,1280, 0,0,0},    //concat
{ "dwconv6", 40,20,1280, 40,20,1280, 3,1,1},    //dwconv6
{ "pwconv6", 40,20,1280, 40,20,96,   1,1,0},    //pwconv6
{ "conv7",   40,20,96,   40,20,10,   1,1,0},    //conv7
};

void Load_IFM(DT32* ifm, DT IBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = Cx*643*323 + (h+h_offset)*643 + (w+w_offset);
                IBUF[c][h][w] = ifm[ifm_index].data[c];
            }
        }
    }
}

void Load_POOL1(DT32* ifm, DT IBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40 + Hx/2;
    int w_offset = Wx*80 + Wx/2;
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = Cx*323*163 + (h+h_offset)*323 + (w+w_offset);
                IBUF[c][h][w] = ifm[ifm_index].data[c];
            }
        }
    }
}

void Load_WBUF3x3(DT32* weight, DT WBUF3x3[32][3][3])
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
            for(int c=0; c<32; c++)
            {
                WBUF3x3[c][m][n] = weight[m*3 + n].data[c];
            }
        }
    }
}

void Load_BBUF(DT32* bias, DT BBUF[32])
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[0].data[c];
    }
}

void Load_WBUF1x1(DT32* weight, DT WBUF1x1[32][32])
{
    for(int m=0; m<32; m++)
    {
        for(int n=0; n<32; n++)
        {
            WBUF1x1[m][n] = weight[m].data[n];
        }
    }
}

void Export_DWCONV1(DT32* ofm, DT OBUF[32][42][82], int Hx, int Wx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    for (int c=0; c<32; c++)
    {
        for (int h=1; h<=40; h++)
        {
            for (int w=1; w<=80; w++)
            {
                int ofm_index = (h+h_offset)*643 + (w+w_offset);
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Export_PWCONV1(DT* ofm, DT OBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40 + Hx/4;
    int w_offset = Wx*80 + Wx/4;
    int c_offset = Cx*32;
    for (int c=0; c<32; c++)
    {
        for (int h=1; h<=40; h++)
        {
            for (int w=1; w<=80; w++)
            {
                int ofm_index = (c+c_offset)*323*643 + (h+h_offset)*643 + (w+w_offset);
                ofm[ofm_index] = OBUF[c][h][w];
            }
        }
    }
}

void Export_POOL1(DT32* ofm, DT OBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*20 + Hx/4;
    int w_offset = Wx*40 + Wx/4;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ofm_index = Cx*163*323 + (h+h_offset)*323 + (w+w_offset);
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Export_DWCONV2(DT32* ofm, DT OBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40 + Hx/2;
    int w_offset = Wx*80 + Wx/2;
    int c_offset = Cx*32;
    for (int c=0; c<32; c++)
    {
        for (int h=1; h<=40; h++)
        {
            for (int w=1; w<=80; w++)
            {
                int ofm_index = Cx*163*323 + (h+h_offset)*323 + (w+w_offset);
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Add_Bias(DT FM[32][42][82], DT BBUF[32], int relu)
{
    for(int h=1; h<=40; h++){
        for(int w=1; w<=80; w++){
            for(int c=0; c<32; c++){
                DT odata = FM[c][h][w];
                odata += BBUF[c];
                if(relu==1)
                {
                    if(odata<0)
                        FM[c][h][w] = 0;
                    else
                        FM[c][h][w] = odata;
                }
            }
        }
    }
}



void Clear_FM(DT FM[32][42][82])
{
    for(int h=0; h<42; h++){
        for(int w=0; w<82; w++){
            for(int c=0; c<32; c++){
                FM[c][h][w] = 0;
            }
        }
    }
}

void Compare(DT FM1[32][42][82], DT FM2[32][42][82])
{
    int error = 0;
    for(int h=1; h<41; h++){
        for(int w=1; w<81; w++){
            for(int c=0; c<32; c++){
                if(abs(FM1[c][h][w]-FM2[c][h][w])>0.001)
                    error++;
                    //printf("FM1[%d][%d][%d]=%f, FM2[%d][%d][%d]=%f\n", c,h,w,FM1[c][h][w],c,h,w,FM2[c][h][w]);
            }
        }
    }
    printf("error count: %d\n", error);
}

void SkyNet_(DT32* ifm, DT32* pool1, DT32* dwconv2, DT32* parameter)
{
    DT FM1[32][42][82]={0};
    DT FM2[32][42][82]={0};
    DT FM3[32][42][82]={0};
    DT FM4[32][42][82]={0};
    DT FM5[32][42][82]={0};

    DT WBUF3x3[4][32][3][3]={0};
    DT WBUF1x1[4][32][32]={0};
    DT BBUF[4][32]={0};

    int weight_offset = 0;
    int bias_offset = 0;

    weight_offset = 96;
    bias_offset = 288;
    Clear_FM(FM4);
    Clear_FM(FM2);
    for(int Hx=0; Hx<4; Hx++)
    {
        for(int Wx=0; Wx<4; Wx++)
        {
            Load_POOL1(pool1, FM1, Hx, Wx, 0);
            Load_POOL1(pool1, FM2, Hx, Wx, 1);

            for(int Mx=0; Mx<3; Mx++)
            {
                printf("Hx: %d, Wx: %d, Mx: %d\n", Hx, Wx, Mx);
                Load_BBUF(parameter + bias_offset + Mx, BBUF[2]);
                printf("load bias from %d\n", bias_offset + Mx);
                Load_WBUF1x1(parameter + weight_offset + Mx*64 + 0, WBUF1x1[0]);
                printf("load weight from %d\n", weight_offset + Mx*64 + 0);
                PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                Load_WBUF1x1(parameter + weight_offset + Mx*64 + 32, WBUF1x1[1]);
                printf("load weight from %d\n", weight_offset +  Mx*64 + 32);
                PWCONV1X1(FM2, FM4, WBUF1x1[1]);
                Add_Bias(FM4, BBUF[2], 1);
                Export_DWCONV2(dwconv2, FM4, Hx, Wx, Mx);
                Clear_FM(FM4);
            }
        }
    }
}









void Load_DWCONV2(DT* ifm, DT IBUF[32][42][82], int Hx, int Wx, int Cx)
{
    int h_offset = Hx*40;
    int w_offset = Wx*80;
    int c_offset = Cx*32;
    for (int h=0; h<40; h++)
    {
        for (int w=0; w<80; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = (c_offset+c)*80*160 + (h+h_offset)*160 + (w+w_offset);
                IBUF[c][h][w] = ifm[ifm_index];
            }
        }
    }
}

void Load_BIAS(DT* bias, DT BBUF[32], int Mx)
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[Mx*32 + c];
    }
}

void load_weight(DT* weight, DT WBUF1x1[32][32], int Nx, int Mx)
{
    for(int m=0; m<32; m++)
    {
        for(int n=0; n<32; n++)
        {
            WBUF1x1[m][n] = weight[(Mx*32 + m)*64 + (Nx*32 + n)];
        }
    }
}

void export_pwconv2e(DT* ofm, DT OBUF[32][42][82], int Hx, int Wx, int Mx)
{
    int h_offset = Hx*40;
    int w_offset = Wx*80;
    int c_offset = Mx*32;
    for (int c=0; c<32; c++)
    {
        for (int h=0; h<40; h++)
        {
            for (int w=0; w<80; w++)
            {
                int ofm_index = (c+c_offset)*80*160 + (h+h_offset)*160 + (w+w_offset);
                ofm[ofm_index] = OBUF[c][h][w];
            }
        }
    }
}

void pwconv11(DT IFM[32][42][82], DT OFM[32][42][82], DT WBUF1x1[32][32])
{
	for(int h=0; h<40; h++){
		for(int w=0; w<80; w++){
			for (int tm=0; tm<32; tm++){
				DT odatatmp = OFM[tm][h][w];
				DT odata = 0;
				for (int tn=0; tn<32; tn++){
					odata += WBUF1x1[tm][tn]*IFM[tn][h][w];
				}
                OFM[tm][h][w] = odata + odatatmp;
			}
		}
	}
}

void AddBias(DT FM[32][42][82], DT BBUF[32], int relu)
{
	for(int h=0; h<40; h++){
		for(int w=0; w<80; w++){
            for(int c=0; c<32; c++){
                DT odata = FM[c][h][w];
                odata += BBUF[c];
                if(relu==1)
                {
                    if(odata<0)
                        FM[c][h][w] = 0;
                    else
                        FM[c][h][w] = odata;
                }
            }
        }
    }
}

void pwconv2e(DT* dwconv2, DT* pwconv2, DT* weight, DT* bias)
{
    DT FM1[32][42][82]={0};
    DT FM2[32][42][82]={0};
    DT FM3[32][42][82]={0};
    DT FM4[32][42][82]={0};
    DT FM5[32][42][82]={0};
    DT WBUF3x3[4][32][3][3]={0};
    DT WBUF1x1[4][32][32]={0};
    DT BBUF[4][32]={0};
    Clear_FM(FM2);
    for(int Hx=0; Hx<2; Hx++)
    {
        for(int Wx=0; Wx<2; Wx++)
        {
            for(int Mx=0; Mx<3; Mx++)
            {
                for(int Nx=0; Nx<2; Nx++)
                {
                    Load_DWCONV2(dwconv2, FM1, Hx, Wx, Nx);
                    load_weight(weight, WBUF1x1[0], Nx, Mx);
                    pwconv11(FM1, FM2, WBUF1x1[0]);
                }
                Load_BIAS(bias, BBUF[0], Mx);
                AddBias(FM2, BBUF[0], 1);
                export_pwconv2e(pwconv2, FM2, Hx, Wx, Mx);
                Clear_FM(FM2);
            }
        }
    }
}


DT32* parameter;
DT* parameter_dt;
DT* data[4];
DT* data_blob;
DT32* data_blob32;
DT* pool1_blob;
DT* pool1[4];
DT32* pool1_blob32;
DT* dwconv2[4];
DT* dwconv2_blob;
DT32* dwconv2_blob32;
DT* pwconv2[4];
DT* pwconv2_blob;
DT32* pwconv2_blob32;
DT* dwconv2_wt;
DT* blob[2];
DT* weight;
DT* bias;
void compare_dt32(DT32* data1, DT32* data2, int len)
{
    int err = 0;
    for(int i=0; i<len; i++)
    {
        for(int j=0; j<32; j++)
        {
            if (((data1[i].data[j] - data2[i].data[j]) > check_scale) || ((data1[i].data[j] - data2[i].data[j]) < -check_scale))
                err++;
        }
    }
    printf("error: %d\n", err);
}

void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        data[p] = (DT*)sds_alloc(32*160*320*sizeof(DT));
        pool1[p] = (DT*)sds_alloc(64*80*160*sizeof(DT));
        dwconv2[p] = (DT*)sds_alloc(64*80*160*sizeof(DT));
        pwconv2[p] = (DT*)sds_alloc(96*80*160*sizeof(DT));
    }
    data_blob = (DT*)sds_alloc(32*323*643*sizeof(DT));
    data_blob32 = (DT32*)sds_alloc(32*323*643*sizeof(DT));
    pool1_blob = (DT*)sds_alloc(64*163*323*sizeof(DT));
    pool1_blob32 = (DT32*)sds_alloc(64*163*323*sizeof(DT));
    dwconv2_blob = (DT*)sds_alloc(64*163*323*sizeof(DT));
    dwconv2_blob32 = (DT32*)sds_alloc(64*163*323*sizeof(DT));
    pwconv2_blob = (DT*)sds_alloc(96*163*323*sizeof(DT));
    pwconv2_blob32 = (DT32*)sds_alloc(96*163*323*sizeof(DT));
    parameter_dt = (DT*)sds_alloc(442634*sizeof(DT));
    parameter = (DT32*)sds_alloc(442634*sizeof(DT));
    blob[0] = (DT*)sds_alloc(3276800*sizeof(DT));
    blob[1] = (DT*)sds_alloc(3276800*sizeof(DT));
    load_weight(parameter, 442634);
    weight = (DT*)sds_alloc(64*96*sizeof(DT));
    bias = (DT*)sds_alloc(96*sizeof(DT));


}

void SkyNet()
{
    load_weight_dt(weight, 96*64);
    load_bias(bias, 96);
    int input = 0;
    load_fm(blob[input], config[4]);
    pwconv2e(blob[input], blob[1-input], weight, bias);
    check_fm(blob[1-input], config[5]);
}