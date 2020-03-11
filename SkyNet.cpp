#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,32, 0,0,0},    //data   0
{ "dwconv1e", 320,160,32, 320,160,32, 3,1,1},    //dwconv1 1
{ "pwconv1e", 320,160,32, 320,160,64, 1,1,0},    //pwconv1 2
{ "pool1e",   320,160,64, 160,80,64,  2,2,0},    //pool1   3
{ "dwconv2e", 160,80,64,  160,80,64,  3,1,1},    //dwconv2 4
{ "pwconv2e", 160,80,64,  160,80,96,  1,1,0},    //pwconv2 5
{ "pool2",   160,80,96,  80,40,96,   2,2,0},    //pool2    6
{ "dwconv3", 80,40,96,   80,40,96,   3,1,1},    //dwconv3  7
{ "pwconv3", 80,40,96,   80,40,192,  1,1,0},    //pwconv3  8
{ "reorg",   80,40,192,  40,20,768,  2,2,0},    //reorg    9
{ "pool3",   80,40,192,  40,20,192,  2,2,0},    //pool3    10
{ "dwconv4", 40,20,192,  40,20,192,  3,1,1},    //dwconv4  11
{ "pwconv4", 40,20,192,  40,20,384,  1,1,0},    //pwconv4  12
{ "dwconv5", 40,20,384,  40,20,384,  3,1,1},    //dwconv5  13 
{ "pwconv5", 40,20,384,  40,20,512,  1,1,0},    //pwconv5  14
{ "cat",     40,20,192,  40,20,1280, 0,0,0},    //concat   15
{ "dwconv6", 40,20,1280, 40,20,1280, 3,1,1},    //dwconv6  16
{ "pwconv6", 40,20,1280, 40,20,96,   1,1,0},    //pwconv6  17
{ "conv7",   40,20,96,   40,20,10,   1,1,0},    //conv7    18
};

void Load_WBUF3x3(DT32* weight, DT WBUF3x3[32][3][3], int Mx, layer l)
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
            for(int c=0; c<32; c++)
            {
                WBUF3x3[c][m][n] = weight[Mx*9 + m*3 + n].data[c];
            }
        }
    }
}

void Load_BBUF(DT* bias, DT BBUF[32], int Mx)
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[Mx*32+c];
    }
}

void Load_FM1(DT* ifm, DT IBUF[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int w=0; w<83; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = (Cx*32+c)*43*83 + h*83 + w;
                IBUF[c][h][w] = ifm[ifm_index];
            }
        }
    }
}

void Add_Bias(DT FM[32][43][83], DT BBUF[32], int relu)
{
    for(int h=0; h<=41; h++){
        for(int w=0; w<=81; w++){
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

void Export_CONV1(DT* ofm, DT OBUF[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int w=0; w<83; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ofm_index = (Cx*32+c)*43*83 + h*83 + w;
                ofm[ofm_index] = OBUF[c][h][w];
            }
        }
    }
}

void Clear_FM(DT FM[32][43][83])
{
    for(int h=0; h<43; h++){
        for(int w=0; w<82; w++){
            for(int c=0; c<32; c++){
                FM[c][h][w] = 0;
            }
        }
    }
}

void dwconv5(DT* pwconv, DT* dwconv, DT32* dwconv_weight, DT* dwconv_bias,  DT* pwconv_weight, DT* pwconv_bias)
{
    DT FM1[32][43][83]={0};
    DT FM2[32][43][83]={0};
    DT FM3[32][43][83]={0};
    DT FM4[32][43][83]={0};
    DT FM5[32][43][83]={0};
    DT WBUF3x3[4][32][3][3]={0};
    DT WBUF1x1[4][32][32]={0};
    DT BBUF[4][32]={0};
    for(int Nx=0; Nx<12; Nx++)
    {
        Load_WBUF3x3(dwconv_weight, WBUF3x3[0], Nx, config[13]);
        Load_BBUF(dwconv_bias, BBUF[0], Nx);
        Load_FM1(pwconv, FM1, Nx);
        DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(dwconv, FM2, Nx);
        Clear_FM(FM2);
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
}

void expand(DT* ifm, DT* ofm, layer l)
{
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ifm_index = c*l.oh*l.ow + h*l.ow + w;
                int ofm_index = c*(l.oh+2)*(l.ow+2) + (h+1)*(l.ow+2) + (w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }    
}

void squeeze(DT* ifm, DT* ofm, layer l)
{
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = c*(l.oh+2)*(l.ow+2) + (h+1)*(l.ow+2) + (w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }   
}

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

DT* pwconv_weight;
DT* pwconv_bias;
DT* dwconv_weight;
DT* dwconv_bias;
DT* dwconv_blob;
DT* pwconv_blob;
DT* data_blob;
DT32* pwconv_weight32;
DT32* pwconv_bias32;
DT32* dwconv_weight32;
DT32* dwconv_bias32;
DT32* dwconv_blob32;
DT32* pwconv_blob32;
DT32* data_blob32;
DT32* paramter;

DT* dwconv[4];
DT* pwconv[4];
DT* data[4];

void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        dwconv[p] = (DT*)sds_alloc(384*43*83*sizeof(DT));
        pwconv[p] = (DT*)sds_alloc(512*43*83*sizeof(DT));
        data[p] = (DT*)sds_alloc(384*43*83*sizeof(DT));
    }

    data_blob = (DT*)sds_alloc(384*83*43*sizeof(DT));
    dwconv_blob = (DT*)sds_alloc(384*83*43*sizeof(DT));
    dwconv_weight = (DT*)sds_alloc(384*9*sizeof(DT));
    dwconv_bias = (DT*)sds_alloc(384*sizeof(DT));
    pwconv_blob = (DT*)sds_alloc(512*43*83*sizeof(DT));
    pwconv_weight = (DT*)sds_alloc(384*512*sizeof(DT));
    pwconv_bias = (DT*)sds_alloc(512*sizeof(DT));
    
    dwconv_blob32 = (DT32*)sds_alloc(384*83*43*sizeof(DT));
    dwconv_weight32 = (DT32*)sds_alloc(384*9*sizeof(DT));
    dwconv_bias32 = (DT32*)sds_alloc(384*sizeof(DT));
    pwconv_blob32 = (DT32*)sds_alloc(512*43*83*sizeof(DT));
    pwconv_weight32 = (DT32*)sds_alloc(384*512*sizeof(DT));
    pwconv_bias32 = (DT32*)sds_alloc(512*sizeof(DT));

    paramter = (DT32*)sds_alloc(442634*sizeof(DT));
}

void SkyNet()
{
    load_bias(dwconv_bias, 384, config[13]);
    load_weight_dt(dwconv_weight, 9*384, config[13]);
    load_bias(pwconv_bias, 512, config[14]);
    load_weight_dt(pwconv_weight, 512*384, config[14]);

    load_weight(paramter, 442634);

    dwconv_w_DT_2_DT32(dwconv_weight, dwconv_weight32, config[13]);
    pwconv_w_DT_2_DT32(pwconv_weight, pwconv_weight32, config[14]);

    int input = 0;
    for(int i=0; i<4; i++)
        load_fm(data[i], config[12]);
    stitch(data, pwconv_blob, config[12]);
    dwconv5(pwconv_blob, dwconv_blob, &paramter[3279], dwconv_bias, pwconv_weight, pwconv_bias);
    distitch(dwconv_blob, dwconv, config[13]);
    for(int i=0; i<4; i++)
        check_fm(dwconv[i], config[13]);
}



/*
    expand(data[0], data[1], config[12]);
    dwconv5(data[1], dwconv[0], &paramter[3279], dwconv_bias, pwconv_weight, pwconv_bias);
    squeeze(dwconv[0], dwconv[1], config[12]);
    check_fm(dwconv[1], config[13]);
    */