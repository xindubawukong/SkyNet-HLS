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
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = (c_offset+c)*82*162 + (h+h_offset)*162 + (w+w_offset);
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

void load_weight11(DT* weight, DT WBUF1x1[32][32], int Nx, int Mx)
{
    for(int m=0; m<32; m++)
    {
        for(int n=0; n<32; n++)
        {
            WBUF1x1[m][n] = weight[(Mx*32 + m)*64 + (Nx*32 + n)];
        }
    }
}

void load_weight33(DT* weight, DT WBUF3x3[32][3][3], int Mx)
{
    for(int c=0; c<32; c++)
    {
        for(int m=0; m<3; m++)
        {
            for(int n=0; n<3; n++)
            {
                WBUF3x3[c][m][n] = weight[(Mx*32+c)*3*3 + m*3 + n];
            }
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
        for (int h=1; h<41; h++)
        {
            for (int w=1; w<81; w++)
            {
                int ofm_index = (c+c_offset)*82*162 + (h+h_offset)*162 + (w+w_offset);
                ofm[ofm_index] = OBUF[c][h][w];
            }
        }
    }
}

void dwconv2e(DT* pool1, DT* dwconv2, DT* pwconv2, DT* dwconv2_weight, DT* dwconv2_bias, DT* pwconv2_weight, DT* pwconv2_bias)
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

            Load_DWCONV2(pool1, FM1, Hx, Wx, 0);
            load_weight33(dwconv2_weight, WBUF3x3[0], 0);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
            Load_BIAS(dwconv2_bias, BBUF[0], 0);
            Add_Bias(FM2, BBUF[0], 1);
            export_pwconv2e(dwconv2, FM2, Hx, Wx, 0);
            

            Load_DWCONV2(pool1, FM1, Hx, Wx, 1);
            load_weight33(dwconv2_weight, WBUF3x3[0], 1);
            DWCONV3X3(FM1, FM3, WBUF3x3[0]);
            Load_BIAS(dwconv2_bias, BBUF[0], 1);
            Add_Bias(FM3, BBUF[0], 1);
            export_pwconv2e(dwconv2, FM3, Hx, Wx, 1);
            
            for(int Mx=0; Mx<3; Mx++)
            {
                load_weight11(pwconv2_weight, WBUF1x1[0], 0, Mx);
                PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                load_weight11(pwconv2_weight, WBUF1x1[0], 1, Mx);
                PWCONV1X1(FM3, FM4, WBUF1x1[0]);

                Load_BIAS(pwconv2_bias, BBUF[0], Mx);
                Add_Bias(FM4, BBUF[0], 1);
                export_pwconv2e(pwconv2, FM4, Hx, Wx, Mx);
                Clear_FM(FM4);
            }
            Clear_FM(FM2);
            Clear_FM(FM3);
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

                Load_DWCONV2(dwconv2, FM1, Hx, Wx, 0);
                load_weight11(weight, WBUF1x1[0], 0, Mx);
                PWCONV1X1(FM1, FM2, WBUF1x1[0]);

                Load_DWCONV2(dwconv2, FM1, Hx, Wx, 1);
                load_weight11(weight, WBUF1x1[0], 1, Mx);
                PWCONV1X1(FM1, FM2, WBUF1x1[0]);

                Load_BIAS(bias, BBUF[0], Mx);
                Add_Bias(FM2, BBUF[0], 1);
                export_pwconv2e(pwconv2, FM2, Hx, Wx, Mx);
                Clear_FM(FM2);
            }
        }
    }
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

DT* dwconv2_weight;
DT* dwconv2_bias;
DT* pwconv2_weight;
DT* pwconv2_bias;
DT32* dwconv2_weight32;
DT32* dwconv2_bias32;
DT32* pwconv2_weight32;
DT32* pwconv2_bias32;
void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        data[p] = (DT*)sds_alloc(32*160*320*sizeof(DT));
        pool1[p] = (DT*)sds_alloc(64*80*160*sizeof(DT));
        dwconv2[p] = (DT*)sds_alloc(64*82*162*sizeof(DT));
        pwconv2[p] = (DT*)sds_alloc(96*82*162*sizeof(DT));
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
    dwconv2_weight = (DT*)sds_alloc(64*9*sizeof(DT));
    dwconv2_bias = (DT*)sds_alloc(64*sizeof(DT));
    pwconv2_weight = (DT*)sds_alloc(64*96*sizeof(DT));
    pwconv2_bias = (DT*)sds_alloc(96*sizeof(DT));
    dwconv2_weight32 = (DT32*)sds_alloc(64*9*sizeof(DT));
    dwconv2_bias32 = (DT32*)sds_alloc(64*sizeof(DT));
    pwconv2_weight32 = (DT32*)sds_alloc(64*96*sizeof(DT));
    pwconv2_bias32 = (DT32*)sds_alloc(96*sizeof(DT));
}

void SkyNet()
{
    load_bias(dwconv2_bias, 64, config[4]);
    load_weight_dt(dwconv2_weight, 9*64, config[4]);
    load_bias(pwconv2_bias, 96, config[5]);
    load_weight_dt(pwconv2_weight, 96*64, config[5]);

    dwconv_w_DT_2_DT32(dwconv2_weight, dwconv2_weight32, config[4]);

    int input = 0;
    load_fm(dwconv2[0], config[3]);
    expand(dwconv2[0], dwconv2[1], config[3]);
    dwconv2e(dwconv2[1], pwconv2[0], pwconv2[2], dwconv2_weight, dwconv2_bias, pwconv2_weight, pwconv2_bias);
    squeeze(pwconv2[0], pwconv2[1], config[4]);
    check_fm(pwconv2[1], config[4]);
    squeeze(pwconv2[2], pwconv2[3], config[5]);
    check_fm(pwconv2[3], config[5]);
}