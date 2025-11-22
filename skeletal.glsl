#version 450
layout(std430, binding = 0) readonly buffer JointMatrices {mat4 mats[];} jointBuf;
layout(push_constant) uniform PC{mat4 proj;mat4 view;mat4 model;mat4 invProj;}pc;
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=2) in uvec4 inJoints;
layout(location=3) in vec4 inWeights;
layout(location=0) out vec3 outColor;
layout(location=1) out vec3 outNormal;
void main(){
   mat4 skinning = (inWeights.x*jointBuf.mats[inJoints.x]) +
                   (inWeights.y*jointBuf.mats[inJoints.y]) +
                   (inWeights.z*jointBuf.mats[inJoints.z]) +
                   (inWeights.w*jointBuf.mats[inJoints.w]);
   gl_Position = pc.proj * pc.view * pc.model * skinning * vec4(inPosition,1.0);
   outNormal = mat3(pc.view * pc.model) * inNormal;
   int v = gl_VertexIndex % 3;
   vec2 uv = v==0 ? vec2(1,0) : v==1 ? vec2(0,1) : vec2(0,0);
   outColor = vec3(uv, 1.0 - uv.x - uv.y);
}