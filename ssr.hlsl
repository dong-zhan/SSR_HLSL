//1, hitPixel is in screen space
//2, rayZMin/rayZMax are in camera space
//3, P0,P1 are in NDC space
//4, depth sampled from depthmap is in [0, 1] NDC space -> so, transform this into camera space is necessary
//5, extra permuted space for P0,P1

// this is derived from the work done by Morgan McGuire and Michael Mara
// their copyright is the following

// By Morgan McGuire and Michael Mara at Williams College 2014
// Released as open source under the BSD 2-Clause License
// http://opensource.org/licenses/BSD-2-Clause
 
//http://casual-effects.blogspot.com/2014/08/screen-space-ray-tracing.html

float linearize_depth(float d,float zNear,float zFar)
{
    float z_n = 2.0 * d - 1.0;
    return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
}

float distanceSquared(float2 a, float2 b) { a -= b; return dot(a, a); }
 
// Returns true if the ray hit something
bool traceScreenSpaceRay(
 // Camera-space ray origin, which must be within the view volume
 float3 csOrig, 
 
 // Unit length camera-space ray direction
 float3 csDir,
 
 // A projection matrix that maps to pixel coordinates (not [-1, +1]
 // normalized device coordinates)
 float4x4 proj, 
 
 // The camera-space Z buffer (all negative values)
 Texture2D csZBuffer,
 
 // Dimensions of csZBuffer
 float2 csZBufferSize,
 
 // Camera space thickness to ascribe to each pixel in the depth buffer
 float zThickness, 
 
 // (Negative number)
 float nearPlaneZ, 
 
 // Step in horizontal or vertical pixels between samples. This is a float
 // because integer math is slow on GPUs, but should be set to an integer >= 1
 float stride,
 
 // Number between 0 and 1 for how far to bump the ray in stride units
 // to conceal banding artifacts
 float jitter,
 
 // Maximum number of iterations. Higher gives better images but may be slow
 const float maxSteps, 
 
 // Maximum camera-space distance to trace before returning a miss
 float maxDistance, 
 
 // Pixel coordinates of the first intersection with the scene
 out float2 hitPixel, 
 
 // Camera space location of the ray hit
 out float3 hitPoint) {
 
    // Clip to the near plane 
    float rayLength = ((csOrig.z + csDir.z * maxDistance) < nearPlaneZ) ?
        (nearPlaneZ-csOrig.z) / csDir.z : maxDistance;		
    float3 csEndPoint = csOrig + csDir * rayLength;
 
    // Project into homogeneous clip space
    float4 H0 = mul(float4(csOrig, 1), proj);			//proj * float4(csOrig, 1.0);
    float4 H1 = mul(float4(csEndPoint, 1), proj);		//proj * float4(csEndPoint, 1.0);
    float k0 = 1.0 / H0.w, k1 = 1.0 / H1.w;
	 
    // The interpolated homogeneous version of the camera-space points   -> the scaled down version?
    float3 Q0 = csOrig * k0, Q1 = csEndPoint * k1;
 
    // Screen-space endpoints
    float2 P0 = H0.xy * k0, P1 = H1.xy * k1;
	

    // If the line is degenerate, make it cover at least one pixel
    // to avoid handling zero-pixel extent as a special case later
    P1 += (distanceSquared(P0, P1) < 0.0001) ? 0.01 : 0.0;
    float2 delta = P1 - P0;


    // Permute so that the primary iteration is in x to collapse
    // all quadrant-specific DDA cases later
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) { 
        // This is a more-vertical line
        permute = true; delta = delta.yx; P0 = P0.yx; P1 = P1.yx; 
    }
 
    float stepDir = sign(delta.x);
    float invdx = stepDir / delta.x;		//
 
    // Track the derivatives of Q and k
    float3  dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;
    float2  dP = float2(stepDir, delta.y * invdx);		//dP = (1, dy/dx)
 
    // Scale derivatives by the desired pixel stride and then
    // offset the starting values by the jitter fraction
    dP *= stride; dQ *= stride; dk *= stride;
    P0 += dP * jitter; Q0 += dQ * jitter; k0 += dk * jitter;
 
    // Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, k from k0 to k1
    float3 Q = Q0; 
 
    // Adjust end condition for iteration direction
    float  end = P1.x * stepDir;

	//use a special case to understand the algorithm, in this case csOrig.z = 10 -> csEnd.z = 1000
 
    float k = k0, stepCount = 0.0, prevZMaxEstimate = csOrig.z;
    float rayZMin = prevZMaxEstimate, rayZMax = prevZMaxEstimate;		//rayZMin/rayZMax is in camera space
    float sceneZMax = rayZMax + 1000;	
	
	float2 P = P0;
	//TODO: depends on the stride, may have to skip more than just one.
	P += dP, Q.z += dQ.z, k += dk;		//skip the first one
	
    for (; 
         ((P.x * stepDir) <= end) && (stepCount < maxSteps) &&
         //((rayZMax > sceneZMax - zThickness) || (rayZMin < sceneZMax)) &&
         ((sceneZMax > rayZMax + zThickness) || (sceneZMax < rayZMin)) &&
          (sceneZMax != 0); 
         P += dP, Q.z += dQ.z, k += dk, ++stepCount) {
         
        rayZMin = prevZMaxEstimate;
        rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
        prevZMaxEstimate = rayZMax;
        if (rayZMin > rayZMax) { 
           float t = rayZMin; rayZMin = rayZMax; rayZMax = t;
        }
 
        hitPixel = permute ? P.yx : P;
        // You may need hitPixel.y = csZBufferSize.y - hitPixel.y; here if your vertical axis
        // is different than ours in screen space
        //sceneZMax = texelFetch(csZBuffer, int2(hitPixel), 0);		//for texelFetch, texture space is used.
					
		hitPixel += 1;
		hitPixel *= 0.5;
		hitPixel.y = 1 - hitPixel.y;
		
		if(hitPixel.y <0 || hitPixel.y >1)return false;
		if(hitPixel.x <0 || hitPixel.x >1)return false;
		
		sceneZMax = csZBuffer.SampleLevel(PointSampler, hitPixel, 0).r;
		//float4 depthViewSample = mul( float4( hitPixelNDC, sceneZMax, 1 ), gPsLastCamInvProj );
		//sceneZMax = depthViewSample.z/depthViewSample.w;
		sceneZMax = linearize_depth(sceneZMax, lastCamNearZ, lastCamFarZ);
		
    }
     
    // Advance Q based on the number of steps
    Q.xy += dQ.xy * stepCount;
    hitPoint = Q * (1.0 / k);
	
    return (sceneZMax <= rayZMax + zThickness) && (sceneZMax > rayZMin);
//    return (rayZMax >= sceneZMax - zThickness) && (rayZMin < sceneZMax);
}


bool getReflectedColorSS(float3 wpos, float3 normal, out float3 color)
{
	float3 inRay = wpos - gPsLastCamPos;
	
	inRay = normalize(inRay);
	float3 R = reflect(inRay, normal);
	
	float4 csOrig = mul( float4( wpos, 1 ), gPsLastCamView );
	float3 csDir = mul( R, (float3x3)gPsLastCamView );

	float zThickness = 0.01f;
	float nearPlaneZ = -0.5f; //gPsCsLastCamPos.z;
	float stride = 22/gPsScreenResolution.y;
	float jitter = 0;
	float maxSteps = 40;
	float maxDistance = 1000;

	float2 hitPixel = 0;
	float3 csHitPoint = 0;

	bool r = traceScreenSpaceRay(csOrig.xyz, csDir, gPsLastCamProj, lastDepthTex, gPsScreenResolution, zThickness, nearPlaneZ, stride, jitter, maxSteps, maxDistance, hitPixel, csHitPoint);
	
	color = lastSceneTex.Sample( LinearWrapSampler, hitPixel ).xyz;

    return r;
}
