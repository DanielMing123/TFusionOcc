#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

static __device__ __forceinline__
void eval_bases24_and_duvw(
    float u, float v, float w,
    float* __restrict__ Bx,  float* __restrict__ By,  float* __restrict__ Bz,
    float* __restrict__ dBxu,float* __restrict__ dBxv,float* __restrict__ dBxw,
    float* __restrict__ dByu,float* __restrict__ dByv,float* __restrict__ dByw,
    float* __restrict__ dBzu,float* __restrict__ dBzv,float* __restrict__ dBzw)
{
    float u2=u*u, v2=v*v, w2=w*w, r2=u2+v2;

    // Zero-init (only to be safe for any we don't overwrite)
    #pragma unroll
    for (int k=0; k<NUM_BASIS; k++){
      dBxu[k]=dBxv[k]=dBxw[k]=0.f;
      dByu[k]=dByv[k]=dByw[k]=0.f;
      dBzu[k]=dBzv[k]=dBzw[k]=0.f;
    }

    // ---- Tier 1
    // B1..B3 constants
    Bx[0]=1.f; By[0]=0.f; Bz[0]=0.f;
    Bx[1]=0.f; By[1]=1.f; Bz[1]=0.f;
    Bx[2]=0.f; By[2]=0.f; Bz[2]=1.f;

    // B4..B6 axis-linear
    Bx[3]=u;   By[3]=0.f; Bz[3]=0.f;   dBxu[3]=1.f;
    Bx[4]=0.f; By[4]=v;   Bz[4]=0.f;   dByv[4]=1.f;
    Bx[5]=0.f; By[5]=0.f; Bz[5]=w;     dBzw[5]=1.f;

    // B7..B9 cross linear
    Bx[6]=v;   By[6]=0.f; Bz[6]=0.f;   dBxv[6]=1.f;
    Bx[7]=w;   By[7]=0.f; Bz[7]=0.f;   dBxw[7]=1.f;
    Bx[8]=0.f; By[8]=w;   Bz[8]=0.f;   dByw[8]=1.f;

    // ---- Tier 2
    // B10..B12 more cross to z
    Bx[9]=0.f;  By[9]=u;   Bz[9]=0.f;  dByu[9]=1.f;
    Bx[10]=0.f; By[10]=0.f;Bz[10]=u;   dBzu[10]=1.f;
    Bx[11]=0.f; By[11]=0.f;Bz[11]=v;   dBzv[11]=1.f;

    // Twists
    // B13 = (-w v, w u, 0)
    Bx[12]=-w*v; By[12]= w*u; Bz[12]=0.f;
    dBxv[12] = -w;  dBxw[12] = -v;
    dByu[12] =  w;  dByw[12] =  u;

    // B14 = (0, -u w, u v)
    Bx[13]=0.f; By[13]= -u*w; Bz[13]= u*v;
    dByu[13] = -w;  dByw[13] = -u;
    dBzu[13] =  v;  dBzv[13] =  u;

    // B15 = (v w, 0, -v u)
    Bx[14]= v*w; By[14]=0.f; Bz[14]= -v*u;
    dBxv[14] = w;  dBxw[14] = v;
    dBzu[14] = -v; dBzv[14] = -u;

    // Bends
    // B16 = (w^2,0,0)
    Bx[15]=w2; By[15]=0.f; Bz[15]=0.f; dBxw[15]=2.f*w;
    // B17 = (0,w^2,0)
    Bx[16]=0.f; By[16]=w2; Bz[16]=0.f; dByw[16]=2.f*w;
    // B18 = (0,0,u^2+v^2)
    Bx[17]=0.f; By[17]=0.f; Bz[17]=r2; dBzu[17]=2.f*u; dBzv[17]=2.f*v;

    // ---- Tier 3
    // Simple quadratics
    // B19=(u^2,0,0)
    Bx[18]=u2; By[18]=0.f; Bz[18]=0.f; dBxu[18]=2.f*u;
    // B20=(0,v^2,0)
    Bx[19]=0.f; By[19]=v2; Bz[19]=0.f; dByv[19]=2.f*v;
    // B21=(0,0,w^2)
    Bx[20]=0.f; By[20]=0.f; Bz[20]=w2; dBzw[20]=2.f*w;

    // B22=(r2*u, r2*v, 0) = (u^3+u v^2, v^3+u^2 v, 0)
    Bx[21]= r2*u;  By[21]= r2*v;  Bz[21]=0.f;
    dBxu[21]= 3.f*u2 + v2;  dBxv[21]= 2.f*u*v;
    dByu[21]= 2.f*u*v;      dByv[21]= 3.f*v2 + u2;

    // B23=(u v, u v, 0)
    Bx[22]= u*v;  By[22]= u*v;  Bz[22]=0.f;
    dBxu[22]= v;  dBxv[22]= u;
    dByu[22]= v;  dByv[22]= u;

    // B24=(softabs(u), softabs(v), 0)
    Bx[23]= u*v2;   By[23]= v*u2;   Bz[23]=0.f;
    dBxu[23] = v * v;        dBxv[23] = 2.f * u * v;   
	dByu[23] = 2.f * u * v;  dByv[23] = u2;         

}

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}


template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int P,
	const int scale_multiplier,
	const uint32_t* __restrict__ offsets,
	const uint32_t* __restrict__ point_list_keys_unsorted,
	const int* __restrict__ voxel2pts,
	const float* __restrict__ pts,
	const float* __restrict__ means3D,
	const float* __restrict__ scales3D,
	const float* __restrict__ rot3D,
	const float* __restrict__ opas,
	const float* __restrict__ u,
	const float* __restrict__ v,
	const float* __restrict__ semantic,
	const float* __restrict__ deform_weight,
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ probability,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
	float* __restrict__ u_grad,
	float* __restrict__ v_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ deform_weights_grad,
	float* __restrict__ rot3D_grad,
	float* __restrict__ scale3D_grad)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
	    return;

	uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t end = offsets[idx];
	const float dof = max(cbrtf(end - start) / scale_multiplier, 1e-6f); // degree of freedom = total samples - 1
	const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float3 rot1 = {rot3D[idx * 9 + 0], rot3D[idx * 9 + 1], rot3D[idx * 9 + 2]};
	const float3 rot2 = {rot3D[idx * 9 + 3], rot3D[idx * 9 + 4], rot3D[idx * 9 + 5]};
	const float3 rot3 = {rot3D[idx * 9 + 6], rot3D[idx * 9 + 7], rot3D[idx * 9 + 8]}; 
	const float3 s = {scales3D[idx * 3], scales3D[idx * 3 + 1], scales3D[idx * 3 + 2]};
	const float opa = opas[idx];
	const float uu = u[idx];
	const float vv = v[idx];
	float sem[CHANNELS] = {0}; // i-th sq's semantic vector
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		sem[ch] = semantic[idx * CHANNELS + ch];
	}
	float dw[NUM_BASIS] = {0}; // i-th sq's deformation weight
	for (int i = 0; i < NUM_BASIS; i++)
	{
		dw[i] = deform_weight[idx * NUM_BASIS + i];
	}
	float means_grad[3] = {0};
	float scales_grad[3] = {0};
	float opa_grad = 0;
	float uu_grad = 0;
	float vv_grad = 0;
	float semantic_grad[CHANNELS] = {0};
	float deform_weight_grad[NUM_BASIS] = {0};
	float rot_grad[9] = {0};

	for (int i = start; i < end; i++)
	{
		int voxel_idx = point_list_keys_unsorted[i];
		int pts_idx = voxel2pts[voxel_idx];
		if (pts_idx >= 0)
		{
			float3 d = {- means.x + pts[pts_idx * 3], - means.y + pts[pts_idx * 3 + 1], - means.z + pts[pts_idx * 3 + 2]};
			float3 ori_trans = {rot1.x * d.x + rot1.y * d.y + rot1.z * d.z, rot2.x * d.x + rot2.y * d.y + rot2.z * d.z, rot3.x * d.x + rot3.y * d.y + rot3.z * d.z}; // trans
			
			/////////// Introduce Inverse-Warp with 24 field-basis /////////
			float3 uvw = {ori_trans.x / s.x, ori_trans.y / s.y, ori_trans.z / s.z}; 
			// Bases + all partials
			float Bx[NUM_BASIS], By[NUM_BASIS], Bz[NUM_BASIS];
			float dBxu[NUM_BASIS], dBxv[NUM_BASIS], dBxw[NUM_BASIS];
			float dByu[NUM_BASIS], dByv[NUM_BASIS], dByw[NUM_BASIS];
			float dBzu[NUM_BASIS], dBzv[NUM_BASIS], dBzw[NUM_BASIS];
			eval_bases24_and_duvw(uvw.x, uvw.y, uvw.z, Bx, By, Bz, dBxu, dBxv, dBxw, dByu, dByv, dByw, dBzu, dBzv, dBzw);
			float3 correc_trans={0.f, 0.f, 0.f};
			for (int k=0; k<NUM_BASIS; k++){
				float wk = dw[k];
				correc_trans.x += wk * Bx[k];
				correc_trans.y += wk * By[k];
				correc_trans.z += wk * Bz[k];
			}
			float3 trans = {ori_trans.x - correc_trans.x, ori_trans.y - correc_trans.y, ori_trans.z - correc_trans.z};
			//////////////////////////////////////////////////////////////////

			float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / uu);
			float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / uu);
			float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / vv);
			float f = powf(term_x + term_y + 1e-6f, uu / vv) + term_z;
			float power = powf(1 + f / dof, -(dof+3)/2);
			// float logp  = -0.5f * (dof + 3.0f) * log1pf(f / dof);
			// float power = expf(logp);
			float prob = power;
			
			float f_grad = 0.;
			float xc_grad = 0.;
			float yc_grad = 0.;
			float zc_grad = 0.;
			float x_grad = 0.;
			float y_grad = 0.;
			float z_grad = 0.;
			float prob_grad = 0.;
			float prob_sum = probability[pts_idx];

			if (prob_sum > 1e-6f) {
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					semantic_grad[ch] += logits_grad[pts_idx * CHANNELS + ch] * prob * opa / prob_sum;
					prob_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * opa / prob_sum;
					opa_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * prob / prob_sum;
				}
			} 
			prob_grad += (1 - bin_logits[pts_idx]) / (1 - power + 1e-6f) *  bin_logits_grad[pts_idx];
			f_grad -= prob_grad * (dof+3) / (2*dof) * powf(1 + f/dof, -(dof+5)/2);
			// f_grad += prob_grad * (- (dof + 3.0f) / (2.0f * (dof + f)) * power);
			uu_grad += f_grad * powf(term_x + term_y + 1e-6f, uu / vv) * ((log(term_x + term_y + 1e-6f) / vv) - (term_x * log((trans.x / s.x) * (trans.x / s.x) + 1e-6f) + term_y * log((trans.y / s.y) * (trans.y / s.y) + 1e-6f)) / uu / vv / (term_x + term_y + 1e-6f));
			vv_grad -= f_grad * (uu * powf(term_x + term_y + 1e-6f, uu / vv) * log(term_x + term_y + 1e-6f) / vv / vv + term_z * log((trans.z / s.z) * (trans.z / s.z) + 1e-6f) / vv / vv);

			scales_grad[0] -= f_grad * 2 * term_x * powf(term_x + term_y + 1e-6f, uu / vv - 1) / vv / s.x;
			scales_grad[1] -= f_grad * 2 * term_y * powf(term_x + term_y + 1e-6f, uu / vv - 1) / vv / s.y;
			scales_grad[2] -= f_grad * 2 * term_z / vv / s.z;

			xc_grad += f_grad * 2 * term_x * powf(term_x + term_y + 1e-6f, uu / vv - 1) / vv / (trans.x + 1e-6f); //trans.x; //xc_grad: corrected Xq,  original is x_grad
			yc_grad += f_grad * 2 * term_y * powf(term_x + term_y + 1e-6f, uu / vv - 1) / vv / (trans.y + 1e-6f); //trans.y;
			zc_grad += f_grad * 2 * term_z / vv / (trans.z + 1e-6f); //trans.z;
			
			///////// Calculate weights wk gradient /////
			for (int k=0; k<NUM_BASIS; k++){
				deform_weight_grad[k] -= (xc_grad * Bx[k] + yc_grad * By[k] + zc_grad * Bz[k]);
			}
			/////////////////////////////////////////////

			////// Sums of basis partials (weighted) for each direction
			float sxu=0.f, syu=0.f, szu=0.f;
			float sxv=0.f, syv=0.f, szv=0.f;
			float sxw=0.f, syw=0.f, bzw=0.f;
			#pragma unroll
			for (int k=0; k<NUM_BASIS; k++){
				float wk = dw[k];
				sxu += wk*dBxu[k];  syu += wk*dByu[k];  szu += wk*dBzu[k];
				sxv += wk*dBxv[k];  syv += wk*dByv[k];  szv += wk*dBzv[k];
				sxw += wk*dBxw[k];  syw += wk*dByw[k];  bzw += wk*dBzw[k];
			}

			scales_grad[0] += (xc_grad * sxu + yc_grad * syu + zc_grad * szu) * ori_trans.x / (s.x*s.x + 1e-6f);
			scales_grad[1] += (xc_grad * sxv + yc_grad * syv + zc_grad * szv) * ori_trans.y / (s.y*s.y + 1e-6f);
			scales_grad[2] += (xc_grad * sxw + yc_grad * syw + zc_grad * bzw) * ori_trans.z / (s.z*s.z + 1e-6f);

			// d(tilde)/dX, dY, dZ
			float dXc_dX = 1.f - sxu/s.x;
			float dYc_dX =     - syu/s.x;
			float dZc_dX =     - szu/s.x;

			float dXc_dY =     - sxv/s.y;
			float dYc_dY = 1.f - syv/s.y;
			float dZc_dY =     - szv/s.y;

			float dXc_dZ =     - sxw/s.z;
			float dYc_dZ =     - syw/s.z;
			float dZc_dZ = 1.f - bzw/s.z;
			x_grad = xc_grad*dXc_dX + yc_grad*dYc_dX + zc_grad*dZc_dX;
			y_grad = xc_grad*dXc_dY + yc_grad*dYc_dY + zc_grad*dZc_dY;
			z_grad = xc_grad*dXc_dZ + yc_grad*dYc_dZ + zc_grad*dZc_dZ;
			/////////////////////////////////////////////////////////////

			means_grad[0] -= (rot1.x * x_grad + rot2.x * y_grad + rot3.x * z_grad);
			means_grad[1] -= (rot1.y * x_grad + rot2.y * y_grad + rot3.y * z_grad);
			means_grad[2] -= (rot1.z * x_grad + rot2.z * y_grad + rot3.z * z_grad);

			rot_grad[0] += x_grad * d.x;
			rot_grad[1] += x_grad * d.y;
			rot_grad[2] += x_grad * d.z;
			rot_grad[3] += y_grad * d.x;
			rot_grad[4] += y_grad * d.y;
			rot_grad[5] += y_grad * d.z;
			rot_grad[6] += z_grad * d.x;
			rot_grad[7] += z_grad * d.y;
			rot_grad[8] += z_grad * d.z;
		}
	}

	means3D_grad[idx * 3] = means_grad[0];
	means3D_grad[idx * 3 + 1] = means_grad[1];
	means3D_grad[idx * 3 + 2] = means_grad[2];

	scale3D_grad[idx * 3] = scales_grad[0];
	scale3D_grad[idx * 3 + 1] = scales_grad[1];
	scale3D_grad[idx * 3 + 2] = scales_grad[2];

	opas_grad[idx] = opa_grad;
	u_grad[idx] = uu_grad;
	v_grad[idx] = vv_grad;
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		semantics_grad[idx * CHANNELS + ch] = semantic_grad[ch];
	}
	for (int k = 0; k < NUM_BASIS; k++)
	{
		deform_weights_grad[idx * NUM_BASIS + k] = deform_weight_grad[k];
	}
	for (int ch = 0; ch < 9; ch++)
	{
		rot3D_grad[idx * 9 + ch] = rot_grad[ch];
	}
}


void BACKWARD::render(
	const int P,
	const int scale_multiplier,
	const uint32_t* offsets,
	const uint32_t* point_list_keys_unsorted,
	const int* voxel2pts,
	const float* pts,
	const float* means3D,
	const float* scales3D,
	const float* rot3D,
	const float* opas,
	const float* u,
	const float* v,
	const float* semantic,
	const float* deform_weight,
	const float* logits,
	const float* bin_logits,
	const float* density,
	const float* probability,
	const float* logits_grad,
	const float* bin_logits_grad,
	const float* density_grad,
	float* means3D_grad,
	float* opas_grad,
	float* u_grad,
	float* v_grad,
	float* semantics_grad,
	float* deform_weights_grad,
	float* rot3D_grad,
	float* scale3D_grad)
{
	renderCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		scale_multiplier,
		offsets,
		point_list_keys_unsorted,
		voxel2pts,
		pts,
		means3D,
		scales3D,
		rot3D,
		opas,
		u,
		v,
		semantic,
		deform_weight,
		logits,
		bin_logits,
		density,
		probability,
		logits_grad,
		bin_logits_grad,
		density_grad,
		means3D_grad,
		opas_grad,
		u_grad,
		v_grad,
		semantics_grad,
		deform_weights_grad,
		rot3D_grad,
		scale3D_grad);
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}