       гK"	  └Ю╫Abrain.Event:2Ъ═╢┼k     FIЦ.	║g▄Ю╫A"▐║
П
)Adam/iterations/Initializer/initial_valueConst*
value	B	 R *"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
з
Adam/iterationsVarHandleOp*
shape: *
	container *"
_class
loc:@Adam/iterations*
_output_shapes
: * 
shared_nameAdam/iterations*
dtype0	
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
Ч
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*"
_class
loc:@Adam/iterations*
dtype0	
П
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
В
!Adam/lr/Initializer/initial_valueConst*
valueB
 *oГ:*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
П
Adam/lrVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/lr*
_output_shapes
: *
shared_name	Adam/lr
_
(Adam/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/lr*
_output_shapes
: 
w
Adam/lr/AssignAssignVariableOpAdam/lr!Adam/lr/Initializer/initial_value*
_class
loc:@Adam/lr*
dtype0
w
Adam/lr/Read/ReadVariableOpReadVariableOpAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
К
%Adam/beta_1/Initializer/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Adam/beta_1*
dtype0
Ы
Adam/beta_1VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/beta_1*
_output_shapes
: *
shared_nameAdam/beta_1
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
З
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
_class
loc:@Adam/beta_1*
dtype0
Г
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1
К
%Adam/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w╛?*
_class
loc:@Adam/beta_2
Ы
Adam/beta_2VarHandleOp*
shape: *
	container *
_class
loc:@Adam/beta_2*
_output_shapes
: *
shared_nameAdam/beta_2*
dtype0
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
З
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
_class
loc:@Adam/beta_2*
dtype0
Г
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
И
$Adam/decay/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Adam/decay
Ш

Adam/decayVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/decay*
_output_shapes
: *
shared_name
Adam/decay
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
Г
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
_class
loc:@Adam/decay*
dtype0
А
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
r
input_1Placeholder*
dtype0*+
_output_shapes
:         * 
shape:         
L
ShapeShapeinput_1*
out_type0*
_output_shapes
:*
T0
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
∙
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
ellipsis_mask *
end_mask *
Index0*
new_axis_mask *
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask
Z
Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
o
Reshape/shapePackstrided_sliceReshape/shape/1*
T0*
N*
_output_shapes
:*

axis 
k
ReshapeReshapeinput_1Reshape/shape*
Tshape0*(
_output_shapes
:         Р*
T0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:*
valueB"     
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *м\▒╜*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *м\▒=*
_class
loc:@dense/kernel
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
_output_shapes
:	Р*
T0
╙
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
з
dense/kernelVarHandleOp*
_output_shapes
: *
shared_namedense/kernel*
dtype0*
shape:	Р*
	container *
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
М
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0
П
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	Р
И
dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
Ь

dense/biasVarHandleOp*
shape:*
	container *
_class
loc:@dense/bias*
_output_shapes
: *
shared_name
dense/bias*
dtype0
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0
Д
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
c
MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р
И
MatMulMatMulReshapeMatMul/ReadVariableOp*'
_output_shapes
:         *
transpose_b( *
transpose_a( *
T0
]
BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
{
BiasAddBiasAddMatMulBiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         
G
ReluReluBiasAdd*'
_output_shapes
:         *
T0
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
shape: *
dtype0
*
_output_shapes
: 
d
cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
I
cond/switch_tIdentitycond/Switch:1*
_output_shapes
: *
T0

G
cond/switch_fIdentitycond/Switch*
_output_shapes
: *
T0

O
cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

f
cond/dropout/rateConst^cond/switch_t*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
m
cond/dropout/ShapeShapecond/dropout/Shape/Switch:1*
out_type0*
_output_shapes
:*
T0
Х
cond/dropout/Shape/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
	loc:@Relu
g
cond/dropout/sub/xConst^cond/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
cond/dropout/subSubcond/dropout/sub/xcond/dropout/rate*
_output_shapes
: *
T0
t
cond/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
t
cond/dropout/random_uniform/maxConst^cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
ж
)cond/dropout/random_uniform/RandomUniformRandomUniformcond/dropout/Shape*
dtype0*'
_output_shapes
:         *
seed2 *

seed *
T0
Й
cond/dropout/random_uniform/subSubcond/dropout/random_uniform/maxcond/dropout/random_uniform/min*
_output_shapes
: *
T0
д
cond/dropout/random_uniform/mulMul)cond/dropout/random_uniform/RandomUniformcond/dropout/random_uniform/sub*
T0*'
_output_shapes
:         
Ц
cond/dropout/random_uniformAddcond/dropout/random_uniform/mulcond/dropout/random_uniform/min*
T0*'
_output_shapes
:         
x
cond/dropout/addAddcond/dropout/subcond/dropout/random_uniform*'
_output_shapes
:         *
T0
_
cond/dropout/FloorFloorcond/dropout/add*'
_output_shapes
:         *
T0
А
cond/dropout/truedivRealDivcond/dropout/Shape/Switch:1cond/dropout/sub*'
_output_shapes
:         *
T0
s
cond/dropout/mulMulcond/dropout/truedivcond/dropout/Floor*'
_output_shapes
:         *
T0
a
cond/IdentityIdentitycond/Identity/Switch*'
_output_shapes
:         *
T0
Р
cond/Identity/SwitchSwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:         :         
q

cond/MergeMergecond/Identitycond/dropout/mul*
T0*
N*)
_output_shapes
:         : 
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *ЇЇї╛*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї>
ы
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
dtype0
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:
*
T0*!
_class
loc:@dense_1/kernel
┌
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
T0
м
dense_1/kernelVarHandleOp*
shape
:
*
	container *!
_class
loc:@dense_1/kernel*
_output_shapes
: *
shared_namedense_1/kernel*
dtype0
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
Ф
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0
Ф
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:

М
dense_1/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

в
dense_1/biasVarHandleOp*
dtype0*
shape:
*
	container *
_class
loc:@dense_1/bias*
_output_shapes
: *
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
Г
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0
К
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

f
MatMul_1/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

П
MatMul_1MatMul
cond/MergeMatMul_1/ReadVariableOp*
transpose_a( *
T0*'
_output_shapes
:         
*
transpose_b( 
a
BiasAdd_1/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_1BiasAddMatMul_1BiasAdd_1/ReadVariableOp*'
_output_shapes
:         
*
T0*
data_formatNHWC
O
SoftmaxSoftmax	BiasAdd_1*
T0*'
_output_shapes
:         

Д
output_1_targetPlaceholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
R
ConstConst*
_output_shapes
:*
valueB*  А?*
dtype0
Д
output_1_sample_weightsPlaceholderWithDefaultConst*
shape:         *
dtype0*#
_output_shapes
:         
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
Й
totalVarHandleOp*
dtype0*
shape: *
	container *
_class

loc:@total*
_output_shapes
: *
shared_nametotal
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
Й
countVarHandleOp*
_class

loc:@count*
_output_shapes
: *
shared_namecount*
dtype0*
shape: *
	container 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
s
 loss/output_1_loss/Reshape/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ф
loss/output_1_loss/ReshapeReshapeoutput_1_target loss/output_1_loss/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0
И
loss/output_1_loss/CastCastloss/output_1_loss/Reshape*
Truncate( *

SrcT0*

DstT0	*#
_output_shapes
:         
s
"loss/output_1_loss/Reshape_1/shapeConst*
_output_shapes
:*
valueB"    
   *
dtype0
Ц
loss/output_1_loss/Reshape_1Reshape	BiasAdd_1"loss/output_1_loss/Reshape_1/shape*
Tshape0*'
_output_shapes
:         
*
T0
У
<loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/output_1_loss/Cast*
_output_shapes
:*
T0	*
out_type0
И
Zloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/output_1_loss/Reshape_1loss/output_1_loss/Cast*
Tlabels0	*6
_output_shapes$
":         :         
*
T0
Ю
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights*
out_type0*
_output_shapes
:*
T0
И
Floss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
р
Floss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
_output_shapes
: *
value	B : *
dtype0
№
Closs/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualEloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
Ж
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
╤
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityQloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
╧
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityOloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

┬
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
э
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

Л
oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualvloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchxloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Ц
vloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchEloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*X
_classN
LJloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
Ъ
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : *
T0
°
iloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Е
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitykloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

Г
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityiloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

И
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

╝
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
valueB :
         *
dtype0
╥
~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЙloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
░
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
М
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЕloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
├
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
:*
valueB"      *
dtype0
┤
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╠
}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*

index_type0*
_output_shapes

:*
T0
п
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
─
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
╛
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
┘
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЛloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*
T0*

Tdim0
┤
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape
С
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЗloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Я
Мloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationАloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:         :         :*
set_operationa-b*
validate_indices(*
T0
╧
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeОloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
е
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
Ы
sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualuloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xДloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
·
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*В
_classx
vtloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
 
hloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergekloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
┬
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergehloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeSloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
з
?loss/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Р
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
Ы
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const**
value!B Boutput_1_sample_weights:0*
dtype0*
_output_shapes
: 
П
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
▐
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*m
valuedBb B\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
М
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Щ
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

╦
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╔
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityLloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
╩
Mloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

г
Jloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Е
Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tK^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: *
T0

М
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
є
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
■
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: **
value!B Boutput_1_sample_weights:0
Є
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB Bvalues.shape=*
dtype0
┴
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*m
valuedBb B\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
я
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
╙
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

В
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

■
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
№
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
ю
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
Й
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fM^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
T0

╢
Kloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeZloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ь
4loss/output_1_loss/broadcast_weights/ones_like/ShapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
╟
4loss/output_1_loss/broadcast_weights/ones_like/ConstConstL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *  А?
т
.loss/output_1_loss/broadcast_weights/ones_likeFill4loss/output_1_loss/broadcast_weights/ones_like/Shape4loss/output_1_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
в
$loss/output_1_loss/broadcast_weightsMuloutput_1_sample_weights.loss/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
═
loss/output_1_loss/MulMulZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits$loss/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0
b
loss/output_1_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Н
loss/output_1_loss/SumSumloss/output_1_loss/Mulloss/output_1_loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
d
loss/output_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Я
loss/output_1_loss/Sum_1Sum$loss/output_1_loss/broadcast_weightsloss/output_1_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
|
loss/output_1_loss/div_no_nanDivNoNanloss/output_1_loss/Sumloss/output_1_loss/Sum_1*
_output_shapes
: *
T0
]
loss/output_1_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
Ш
loss/output_1_loss/MeanMeanloss/output_1_loss/div_no_nanloss/output_1_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_1_loss/Mean*
_output_shapes
: *
T0
Г
metrics/acc/CastCastoutput_1_target*
Truncate( *

SrcT0*

DstT0*0
_output_shapes
:                  
~
metrics/acc/SqueezeSqueezemetrics/acc/Cast*#
_output_shapes
:         *
T0*
squeeze_dims

         
g
metrics/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Р
metrics/acc/ArgMaxArgMaxSoftmaxmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
{
metrics/acc/Cast_1Castmetrics/acc/ArgMax*

DstT0*#
_output_shapes
:         *
Truncate( *

SrcT0	
q
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast_1*#
_output_shapes
:         *
T0
z
metrics/acc/Cast_2Castmetrics/acc/Equal*

DstT0*#
_output_shapes
:         *
Truncate( *

SrcT0

]
metrics/acc/SizeSizemetrics/acc/Cast_2*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_3Castmetrics/acc/Size*
Truncate( *

SrcT0*

DstT0*
_output_shapes
: 
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/SumSummetrics/acc/Cast_2metrics/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
z
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
}
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_3^metrics/acc/ReadVariableOp*
dtype0
Ы
metrics/acc/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1^metrics/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
В
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal^metrics/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Д
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount^metrics/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0

metrics/acc/Squeeze_1Squeezeoutput_1_target*
T0*
squeeze_dims

         *#
_output_shapes
:         
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
metrics/acc/ArgMax_1ArgMaxSoftmaxmetrics/acc/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:         *

Tidx0*
T0
}
metrics/acc/Cast_4Castmetrics/acc/ArgMax_1*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
u
metrics/acc/Equal_1Equalmetrics/acc/Squeeze_1metrics/acc/Cast_4*#
_output_shapes
:         *
T0
|
metrics/acc/Cast_5Castmetrics/acc/Equal_1*

DstT0*#
_output_shapes
:         *
Truncate( *

SrcT0

]
metrics/acc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

metrics/acc/MeanMeanmetrics/acc/Cast_5metrics/acc/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
valueB *
_class
loc:@loss/mul*
dtype0*
_output_shapes
: 
Г
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?*
_class
loc:@loss/mul
╢
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: *
T0
е
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_1_loss/Mean*
_output_shapes
: *
T0*
_class
loc:@loss/mul
Ъ
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
▒
Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB **
_class 
loc:@loss/output_1_loss/Mean*
dtype0*
_output_shapes
: 
У
<training/Adam/gradients/loss/output_1_loss/Mean_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shape*
T0*
Tshape0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
й
:training/Adam/gradients/loss/output_1_loss/Mean_grad/ConstConst*
_output_shapes
: *
valueB **
_class 
loc:@loss/output_1_loss/Mean*
dtype0
Ъ
9training/Adam/gradients/loss/output_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape:training/Adam/gradients/loss/output_1_loss/Mean_grad/Const*
T0*

Tmultiples0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1Const*
valueB
 *  А?**
_class 
loc:@loss/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/output_1_loss/Mean_grad/Tile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
╡
@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB *0
_class&
$"loc:@loss/output_1_loss/div_no_nan
╖
Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1Const*
valueB *0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
▐
Ptraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgs@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ShapeBtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*2
_output_shapes 
:         :         
№
Etraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivloss/output_1_loss/Sum_1*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╬
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/SumSumEtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nanPtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
░
Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ReshapeReshape>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape*
Tshape0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
░
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/NegNegloss/output_1_loss/Sum*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: 
А
Gtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNan>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Negloss/output_1_loss/Sum_1*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: 
Й
Gtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanGtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_1loss/output_1_loss/Sum_1*
_output_shapes
: *
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan
Я
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/mulMul<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivGtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_2*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╦
@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum_1Sum>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/mulRtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan
╢
Dtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Reshape_1Reshape@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum_1Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1*
Tshape0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╢
Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape/shapeConst*)
_class
loc:@loss/output_1_loss/Sum*
dtype0*
_output_shapes
:*
valueB:
л
;training/Adam/gradients/loss/output_1_loss/Sum_grad/ReshapeReshapeBtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ReshapeAtraining/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
:
║
9training/Adam/gradients/loss/output_1_loss/Sum_grad/ShapeShapeloss/output_1_loss/Mul*
T0*
out_type0*)
_class
loc:@loss/output_1_loss/Sum*
_output_shapes
:
г
8training/Adam/gradients/loss/output_1_loss/Sum_grad/TileTile;training/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape*#
_output_shapes
:         *
T0*

Tmultiples0*)
_class
loc:@loss/output_1_loss/Sum
■
9training/Adam/gradients/loss/output_1_loss/Mul_grad/ShapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/output_1_loss/Mul
╩
;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1Shape$loss/output_1_loss/broadcast_weights*
out_type0*)
_class
loc:@loss/output_1_loss/Mul*
_output_shapes
:*
T0
┬
Itraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1*
T0*)
_class
loc:@loss/output_1_loss/Mul*2
_output_shapes 
:         :         
ў
7training/Adam/gradients/loss/output_1_loss/Mul_grad/MulMul8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile$loss/output_1_loss/broadcast_weights*
T0*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         
н
7training/Adam/gradients/loss/output_1_loss/Mul_grad/SumSum7training/Adam/gradients/loss/output_1_loss/Mul_grad/MulItraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgs*
T0*)
_class
loc:@loss/output_1_loss/Mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
б
;training/Adam/gradients/loss/output_1_loss/Mul_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum9training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape*
T0*
Tshape0*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         
п
9training/Adam/gradients/loss/output_1_loss/Mul_grad/Mul_1MulZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         *
T0
│
9training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum_1Sum9training/Adam/gradients/loss/output_1_loss/Mul_grad/Mul_1Ktraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/output_1_loss/Mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
з
=training/Adam/gradients/loss/output_1_loss/Mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum_1;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1*
T0*
Tshape0*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         
о
"training/Adam/gradients/zeros_like	ZerosLike\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╤
Зtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
┴
Жtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0
Д
Вtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/output_1_loss/Mul_grad/ReshapeЖtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:         *
T0*

Tdim0*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
▓
{training/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulВtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsЗtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
*
T0
╣
?training/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_1*
T0*
out_type0*/
_class%
#!loc:@loss/output_1_loss/Reshape_1*
_output_shapes
:
√
Atraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ReshapeReshape{training/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul?training/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Shape*
Tshape0*/
_class%
#!loc:@loss/output_1_loss/Reshape_1*'
_output_shapes
:         
*
T0
▐
2training/Adam/gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@BiasAdd_1*
_output_shapes
:
*
T0*
data_formatNHWC
З
,training/Adam/gradients/MatMul_1_grad/MatMulMatMulAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ReshapeMatMul_1/ReadVariableOp*
_class
loc:@MatMul_1*'
_output_shapes
:         *
transpose_b(*
T0*
transpose_a( 
є
.training/Adam/gradients/MatMul_1_grad/MatMul_1MatMul
cond/MergeAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Reshape*
_output_shapes

:
*
transpose_b( *
T0*
transpose_a(*
_class
loc:@MatMul_1
┘
1training/Adam/gradients/cond/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_1_grad/MatMulcond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@MatMul_1
м
3training/Adam/gradients/cond/dropout/mul_grad/ShapeShapecond/dropout/truediv*
T0*
out_type0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
м
5training/Adam/gradients/cond/dropout/mul_grad/Shape_1Shapecond/dropout/Floor*
out_type0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:*
T0
к
Ctraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/mul_grad/Shape5training/Adam/gradients/cond/dropout/mul_grad/Shape_1*
T0*#
_class
loc:@cond/dropout/mul*2
_output_shapes 
:         :         
╪
1training/Adam/gradients/cond/dropout/mul_grad/MulMul3training/Adam/gradients/cond/Merge_grad/cond_grad:1cond/dropout/Floor*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:         *
T0
Х
1training/Adam/gradients/cond/dropout/mul_grad/SumSum1training/Adam/gradients/cond/dropout/mul_grad/MulCtraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs*#
_class
loc:@cond/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Н
5training/Adam/gradients/cond/dropout/mul_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/mul_grad/Sum3training/Adam/gradients/cond/dropout/mul_grad/Shape*
T0*
Tshape0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:         
▄
3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Mulcond/dropout/truediv3training/Adam/gradients/cond/Merge_grad/cond_grad:1*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:         *
T0
Ы
3training/Adam/gradients/cond/dropout/mul_grad/Sum_1Sum3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Etraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*#
_class
loc:@cond/dropout/mul
У
7training/Adam/gradients/cond/dropout/mul_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/mul_grad/Sum_15training/Adam/gradients/cond/dropout/mul_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0*#
_class
loc:@cond/dropout/mul
Ъ
training/Adam/gradients/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
	loc:@Relu
Щ
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*
_class
	loc:@Relu*'
_output_shapes
:         *
T0
Ш
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
_class
	loc:@Relu*
_output_shapes
:*
T0*
out_type0
д
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
_output_shapes
: *
valueB
 *    *
_class
	loc:@Relu*
dtype0
╚
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*

index_type0*
_class
	loc:@Relu*'
_output_shapes
:         
ь
;training/Adam/gradients/cond/Identity/Switch_grad/cond_gradMerge1training/Adam/gradients/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
N*
T0*
_class
	loc:@Relu*)
_output_shapes
:         : 
╗
7training/Adam/gradients/cond/dropout/truediv_grad/ShapeShapecond/dropout/Shape/Switch:1*
out_type0*'
_class
loc:@cond/dropout/truediv*
_output_shapes
:*
T0
е
9training/Adam/gradients/cond/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *'
_class
loc:@cond/dropout/truediv*
dtype0
║
Gtraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/cond/dropout/truediv_grad/Shape9training/Adam/gradients/cond/dropout/truediv_grad/Shape_1*'
_class
loc:@cond/dropout/truediv*2
_output_shapes 
:         :         *
T0
ш
9training/Adam/gradients/cond/dropout/truediv_grad/RealDivRealDiv5training/Adam/gradients/cond/dropout/mul_grad/Reshapecond/dropout/sub*'
_output_shapes
:         *
T0*'
_class
loc:@cond/dropout/truediv
й
5training/Adam/gradients/cond/dropout/truediv_grad/SumSum9training/Adam/gradients/cond/dropout/truediv_grad/RealDivGtraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*'
_class
loc:@cond/dropout/truediv
Э
9training/Adam/gradients/cond/dropout/truediv_grad/ReshapeReshape5training/Adam/gradients/cond/dropout/truediv_grad/Sum7training/Adam/gradients/cond/dropout/truediv_grad/Shape*
T0*
Tshape0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
┤
5training/Adam/gradients/cond/dropout/truediv_grad/NegNegcond/dropout/Shape/Switch:1*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
ъ
;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_1RealDiv5training/Adam/gradients/cond/dropout/truediv_grad/Negcond/dropout/sub*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
Ё
;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_2RealDiv;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_1cond/dropout/sub*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         *
T0
Л
5training/Adam/gradients/cond/dropout/truediv_grad/mulMul5training/Adam/gradients/cond/dropout/mul_grad/Reshape;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_2*'
_output_shapes
:         *
T0*'
_class
loc:@cond/dropout/truediv
й
7training/Adam/gradients/cond/dropout/truediv_grad/Sum_1Sum5training/Adam/gradients/cond/dropout/truediv_grad/mulItraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*'
_class
loc:@cond/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Т
;training/Adam/gradients/cond/dropout/truediv_grad/Reshape_1Reshape7training/Adam/gradients/cond/dropout/truediv_grad/Sum_19training/Adam/gradients/cond/dropout/truediv_grad/Shape_1*
Tshape0*'
_class
loc:@cond/dropout/truediv*
_output_shapes
: *
T0
Ь
 training/Adam/gradients/Switch_1SwitchRelucond/pred_id*
T0*
_class
	loc:@Relu*:
_output_shapes(
&:         :         
Ы
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
_class
	loc:@Relu*'
_output_shapes
:         *
T0
Ш
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
T0*
out_type0*
_class
	loc:@Relu*
_output_shapes
:
и
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
valueB
 *    *
_class
	loc:@Relu*
dtype0*
_output_shapes
: 
╠
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*

index_type0*
_class
	loc:@Relu*'
_output_shapes
:         *
T0
√
@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_19training/Adam/gradients/cond/dropout/truediv_grad/Reshape*)
_output_shapes
:         : *
N*
T0*
_class
	loc:@Relu
ў
training/Adam/gradients/AddNAddN;training/Adam/gradients/cond/Identity/Switch_grad/cond_grad@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_grad*
N*
T0*
_class
	loc:@Relu*'
_output_shapes
:         
е
*training/Adam/gradients/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddNRelu*
_class
	loc:@Relu*'
_output_shapes
:         *
T0
├
0training/Adam/gradients/BiasAdd_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_class
loc:@BiasAdd*
_output_shapes
:
ы
*training/Adam/gradients/MatMul_grad/MatMulMatMul*training/Adam/gradients/Relu_grad/ReluGradMatMul/ReadVariableOp*(
_output_shapes
:         Р*
transpose_b(*
T0*
transpose_a( *
_class
loc:@MatMul
╓
,training/Adam/gradients/MatMul_grad/MatMul_1MatMulReshape*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
transpose_a(*
_class
loc:@MatMul*
_output_shapes
:	Р*
transpose_b( 
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
И
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
_output_shapes
: *
dtype0	
И
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations^training/Adam/ReadVariableOp*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*
Truncate( *

SrcT0	*

DstT0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
o
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  А
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
T0*
_output_shapes
: 
Г
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
s
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/Cast*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/ReadVariableOp_1ReadVariableOpAdam/lr*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
T0*
_output_shapes
: 
t
#training/Adam/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"     *
dtype0
^
training/Adam/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ч
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
_output_shapes
:	Р*
T0*

index_type0
┼
training/Adam/VariableVarHandleOp*
shape:	Р*
	container *)
_class
loc:@training/Adam/Variable*
_output_shapes
: *'
shared_nametraining/Adam/Variable*
dtype0
}
7training/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
Ц
training/Adam/Variable/AssignAssignVariableOptraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
dtype0
н
*training/Adam/Variable/Read/ReadVariableOpReadVariableOptraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
:	Р
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
╞
training/Adam/Variable_1VarHandleOp*
dtype0*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: *)
shared_nametraining/Adam/Variable_1
Б
9training/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
Ю
training/Adam/Variable_1/AssignAssignVariableOptraining/Adam/Variable_1training/Adam/zeros_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0
о
,training/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
:
j
training/Adam/zeros_2Const*
_output_shapes

:
*
valueB
*    *
dtype0
╩
training/Adam/Variable_2VarHandleOp*
shape
:
*
	container *+
_class!
loc:@training/Adam/Variable_2*
_output_shapes
: *)
shared_nametraining/Adam/Variable_2*
dtype0
Б
9training/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
Ю
training/Adam/Variable_2/AssignAssignVariableOptraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0
▓
,training/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes

:

b
training/Adam/zeros_3Const*
valueB
*    *
dtype0*
_output_shapes
:

╞
training/Adam/Variable_3VarHandleOp*
dtype0*
shape:
*
	container *+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: *)
shared_nametraining/Adam/Variable_3
Б
9training/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
Ю
training/Adam/Variable_3/AssignAssignVariableOptraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0
о
,training/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_3
v
%training/Adam/zeros_4/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Э
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes
:	Р
╦
training/Adam/Variable_4VarHandleOp*
shape:	Р*
	container *+
_class!
loc:@training/Adam/Variable_4*
_output_shapes
: *)
shared_nametraining/Adam/Variable_4*
dtype0
Б
9training/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
Ю
training/Adam/Variable_4/AssignAssignVariableOptraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0
│
,training/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
_output_shapes
:	Р*+
_class!
loc:@training/Adam/Variable_4*
dtype0
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
╞
training/Adam/Variable_5VarHandleOp*
dtype0*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: *)
shared_nametraining/Adam/Variable_5
Б
9training/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
Ю
training/Adam/Variable_5/AssignAssignVariableOptraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0
о
,training/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_5*
dtype0
j
training/Adam/zeros_6Const*
dtype0*
_output_shapes

:
*
valueB
*    
╩
training/Adam/Variable_6VarHandleOp*
dtype0*
shape
:
*
	container *+
_class!
loc:@training/Adam/Variable_6*
_output_shapes
: *)
shared_nametraining/Adam/Variable_6
Б
9training/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
Ю
training/Adam/Variable_6/AssignAssignVariableOptraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0
▓
,training/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes

:

b
training/Adam/zeros_7Const*
valueB
*    *
dtype0*
_output_shapes
:

╞
training/Adam/Variable_7VarHandleOp*
dtype0*
shape:
*
	container *+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
: *)
shared_nametraining/Adam/Variable_7
Б
9training/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
Ю
training/Adam/Variable_7/AssignAssignVariableOptraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0
о
,training/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_7*
dtype0
o
%training/Adam/zeros_8/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*
_output_shapes
:
╞
training/Adam/Variable_8VarHandleOp*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: *)
shared_nametraining/Adam/Variable_8*
dtype0
Б
9training/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
Ю
training/Adam/Variable_8/AssignAssignVariableOptraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0
о
,training/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
dtype0*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_8
o
%training/Adam/zeros_9/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
`
training/Adam/zeros_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
training/Adam/zeros_9Fill%training/Adam/zeros_9/shape_as_tensortraining/Adam/zeros_9/Const*

index_type0*
_output_shapes
:*
T0
╞
training/Adam/Variable_9VarHandleOp*
dtype0*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
: *)
shared_nametraining/Adam/Variable_9
Б
9training/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
Ю
training/Adam/Variable_9/AssignAssignVariableOptraining/Adam/Variable_9training/Adam/zeros_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0
о
,training/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_9*
dtype0*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_9
p
&training/Adam/zeros_10/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
_output_shapes
:*
T0*

index_type0
╔
training/Adam/Variable_10VarHandleOp*
_output_shapes
: **
shared_nametraining/Adam/Variable_10*
dtype0*
shape:*
	container *,
_class"
 loc:@training/Adam/Variable_10
Г
:training/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
в
 training/Adam/Variable_10/AssignAssignVariableOptraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0
▒
-training/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_10*
dtype0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_10
p
&training/Adam/zeros_11/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_11Fill&training/Adam/zeros_11/shape_as_tensortraining/Adam/zeros_11/Const*
T0*

index_type0*
_output_shapes
:
╔
training/Adam/Variable_11VarHandleOp*
dtype0*
shape:*
	container *,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: **
shared_nametraining/Adam/Variable_11
Г
:training/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
в
 training/Adam/Variable_11/AssignAssignVariableOptraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0
▒
-training/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
z
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Р
И
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
_output_shapes
:	Р*
T0
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
Z
training/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
_output_shapes
: *
T0
З
training/Adam/mul_2Multraining/Adam/sub_2,training/Adam/gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	Р
l
training/Adam/addAddtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes
:	Р
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
|
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
И
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
T0*
_output_shapes
:	Р
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
_output_shapes
: *
T0
v
training/Adam/SquareSquare,training/Adam/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	Р*
T0
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes
:	Р*
T0
n
training/Adam/add_1Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes
:	Р
j
training/Adam/mul_5Multraining/Adam/multraining/Adam/add*
_output_shapes
:	Р*
T0
Z
training/Adam/Const_3Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Ж
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_1training/Adam/Const_4*
_output_shapes
:	Р*
T0
Р
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*
T0*
_output_shapes
:	Р
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes
:	Р*
T0
Z
training/Adam/add_2/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
q
training/Adam/add_2Addtraining/Adam/Sqrt_1training/Adam/add_2/y*
_output_shapes
:	Р*
T0
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_2*
_output_shapes
:	Р*
T0
l
training/Adam/ReadVariableOp_6ReadVariableOpdense/kernel*
_output_shapes
:	Р*
dtype0
}
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
_output_shapes
:	Р*
T0
j
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/Variabletraining/Adam/add*
dtype0
Ч
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/Variable^training/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Р
p
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/Variable_4training/Adam/add_1*
dtype0
Ы
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/Variable_4!^training/Adam/AssignVariableOp_1*
dtype0*
_output_shapes
:	Р
d
 training/Adam/AssignVariableOp_2AssignVariableOpdense/kerneltraining/Adam/sub_4*
dtype0
П
training/Adam/ReadVariableOp_9ReadVariableOpdense/kernel!^training/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Р
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
dtype0*
_output_shapes
:
Д
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
Ж
training/Adam/mul_7Multraining/Adam/sub_50training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
i
training/Adam/add_3Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes
:
Д
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
_output_shapes
: *
T0
w
training/Adam/Square_1Square0training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_4Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_3*
T0*
_output_shapes
:
Z
training/Adam/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_6Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Б
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_4training/Adam/Const_6*
_output_shapes
:*
T0
Л
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
T0*
_output_shapes
:
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:
Z
training/Adam/add_5/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
l
training/Adam/add_5Addtraining/Adam/Sqrt_2training/Adam/add_5/y*
_output_shapes
:*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_5*
_output_shapes
:*
T0
f
training/Adam/ReadVariableOp_14ReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
y
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
T0*
_output_shapes
:
p
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/Variable_1training/Adam/add_3*
dtype0
Ч
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/Variable_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:
p
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/Variable_5training/Adam/add_4*
dtype0
Ч
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/Variable_5!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
:
b
 training/Adam/AssignVariableOp_5AssignVariableOp
dense/biastraining/Adam/sub_7*
dtype0
Й
training/Adam/ReadVariableOp_17ReadVariableOp
dense/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/Variable_2*
dtype0*
_output_shapes

:

К
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
_output_shapes

:
*
T0
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
_output_shapes
: *
T0
Й
training/Adam/mul_12Multraining/Adam/sub_8.training/Adam/gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_6Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:
*
T0
c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/Variable_6*
_output_shapes

:
*
dtype0
К
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
_output_shapes

:
*
T0
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
Z
training/Adam/sub_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
y
training/Adam/Square_2Square.training/Adam/gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
*
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
_output_shapes

:
*
T0
o
training/Adam/add_7Addtraining/Adam/mul_13training/Adam/mul_14*
_output_shapes

:
*
T0
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_6*
_output_shapes

:
*
T0
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Е
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_7training/Adam/Const_8*
T0*
_output_shapes

:

П
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
_output_shapes

:
*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
_output_shapes

:
*
T0
Z
training/Adam/add_8/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
p
training/Adam/add_8Addtraining/Adam/Sqrt_3training/Adam/add_8/y*
T0*
_output_shapes

:

v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_8*
_output_shapes

:
*
T0
n
training/Adam/ReadVariableOp_22ReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
~
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
T0*
_output_shapes

:

p
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/Variable_2training/Adam/add_6*
dtype0
Ы
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/Variable_2!^training/Adam/AssignVariableOp_6*
dtype0*
_output_shapes

:

p
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/Variable_6training/Adam/add_7*
dtype0
Ы
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/Variable_6!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes

:

g
 training/Adam/AssignVariableOp_8AssignVariableOpdense_1/kerneltraining/Adam/sub_10*
dtype0
С
training/Adam/ReadVariableOp_25ReadVariableOpdense_1/kernel!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes

:

c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
dtype0*
_output_shapes
:

Ж
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
T0*
_output_shapes
:

c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
К
training/Adam/mul_17Multraining/Adam/sub_112training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:

k
training/Adam/add_9Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
:
*
T0
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:

Ж
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes
:
*
T0
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
y
training/Adam/Square_3Square2training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:
*
T0
l
training/Adam/add_10Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:

h
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_9*
T0*
_output_shapes
:

Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Г
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_10training/Adam/Const_10*
_output_shapes
:
*
T0
Л
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
_output_shapes
:
*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
:
*
T0
[
training/Adam/add_11/yConst*
dtype0*
_output_shapes
: *
valueB
 *Х┐╓3
n
training/Adam/add_11Addtraining/Adam/Sqrt_4training/Adam/add_11/y*
_output_shapes
:
*
T0
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_11*
T0*
_output_shapes
:

h
training/Adam/ReadVariableOp_30ReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
z
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
_output_shapes
:
*
T0
p
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/Variable_3training/Adam/add_9*
dtype0
Ч
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/Variable_3!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:

r
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/Variable_7training/Adam/add_10*
dtype0
Ш
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/Variable_7"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:

f
!training/Adam/AssignVariableOp_11AssignVariableOpdense_1/biastraining/Adam/sub_13*
dtype0
М
training/Adam/ReadVariableOp_33ReadVariableOpdense_1/bias"^training/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
:

╓
training_1/group_depsNoOp	^loss/mul^metrics/acc/div_no_nan ^training/Adam/ReadVariableOp_15 ^training/Adam/ReadVariableOp_16 ^training/Adam/ReadVariableOp_17 ^training/Adam/ReadVariableOp_23 ^training/Adam/ReadVariableOp_24 ^training/Adam/ReadVariableOp_25 ^training/Adam/ReadVariableOp_31 ^training/Adam/ReadVariableOp_32 ^training/Adam/ReadVariableOp_33^training/Adam/ReadVariableOp_7^training/Adam/ReadVariableOp_8^training/Adam/ReadVariableOp_9
Z
VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
Adam/decay*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense/kernel*
_output_shapes
: 
\
VarIsInitializedOp_3VarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
\
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
\
VarIsInitializedOp_5VarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
\
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
R
VarIsInitializedOp_7VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
]
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
S
VarIsInitializedOp_9VarIsInitializedOpAdam/iterations*
_output_shapes
: 
Q
VarIsInitializedOp_10VarIsInitializedOpdense_1/bias*
_output_shapes
: 
J
VarIsInitializedOp_11VarIsInitializedOpcount*
_output_shapes
: 
J
VarIsInitializedOp_12VarIsInitializedOptotal*
_output_shapes
: 
^
VarIsInitializedOp_13VarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
P
VarIsInitializedOp_14VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
]
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
]
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
O
VarIsInitializedOp_17VarIsInitializedOp
dense/bias*
_output_shapes
: 
L
VarIsInitializedOp_18VarIsInitializedOpAdam/lr*
_output_shapes
: 
P
VarIsInitializedOp_19VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
[
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
]
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
]
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
В
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^count/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^total/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
E
AssignVariableOpAssignVariableOptotalPlaceholder*
dtype0
_
ReadVariableOpReadVariableOptotal^AssignVariableOp*
dtype0*
_output_shapes
: 
N
Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 
I
AssignVariableOp_1AssignVariableOpcountPlaceholder_1*
dtype0
c
ReadVariableOp_1ReadVariableOpcount^AssignVariableOp_1*
dtype0*
_output_shapes
: 
A
evaluation/group_depsNoOp	^loss/mul^metrics/acc/div_no_nan
Н
(SGD/iterations/Initializer/initial_valueConst*
value	B	 R *!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
д
SGD/iterationsVarHandleOp*
dtype0	*
shape: *
	container *!
_class
loc:@SGD/iterations*
_output_shapes
: *
shared_nameSGD/iterations
m
/SGD/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/iterations*
_output_shapes
: 
У
SGD/iterations/AssignAssignVariableOpSGD/iterations(SGD/iterations/Initializer/initial_value*!
_class
loc:@SGD/iterations*
dtype0	
М
"SGD/iterations/Read/ReadVariableOpReadVariableOpSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
А
 SGD/lr/Initializer/initial_valueConst*
valueB
 *
╫#<*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
М
SGD/lrVarHandleOp*
shape: *
	container *
_class
loc:@SGD/lr*
_output_shapes
: *
shared_nameSGD/lr*
dtype0
]
'SGD/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/lr*
_output_shapes
: 
s
SGD/lr/AssignAssignVariableOpSGD/lr SGD/lr/Initializer/initial_value*
_class
loc:@SGD/lr*
dtype0
t
SGD/lr/Read/ReadVariableOpReadVariableOpSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
М
&SGD/momentum/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@SGD/momentum
Ю
SGD/momentumVarHandleOp*
shape: *
	container *
_class
loc:@SGD/momentum*
_output_shapes
: *
shared_nameSGD/momentum*
dtype0
i
-SGD/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/momentum*
_output_shapes
: 
Л
SGD/momentum/AssignAssignVariableOpSGD/momentum&SGD/momentum/Initializer/initial_value*
_class
loc:@SGD/momentum*
dtype0
Ж
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
Ж
#SGD/decay/Initializer/initial_valueConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@SGD/decay*
dtype0
Х
	SGD/decayVarHandleOp*
shape: *
	container *
_class
loc:@SGD/decay*
_output_shapes
: *
shared_name	SGD/decay*
dtype0
c
*SGD/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp	SGD/decay*
_output_shapes
: 

SGD/decay/AssignAssignVariableOp	SGD/decay#SGD/decay/Initializer/initial_value*
_class
loc:@SGD/decay*
dtype0
}
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
t
	input_1_1Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
P
Shape_1Shape	input_1_1*
T0*
out_type0*
_output_shapes
:
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Г
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
new_axis_mask *
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
ellipsis_mask *
end_mask 
\
Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
         
u
Reshape_1/shapePackstrided_slice_1Reshape_1/shape/1*
_output_shapes
:*

axis *
T0*
N
q
	Reshape_1Reshape	input_1_1Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:         Р
г
/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"     *!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
:
Х
-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *м\▒╜*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
Х
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *м\▒=
ь
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Р*
dtype0
╓
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
T0
щ
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Р*
T0*!
_class
loc:@dense_2/kernel
█
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Р*
T0
н
dense_2/kernelVarHandleOp*
shape:	Р*
	container *!
_class
loc:@dense_2/kernel*
_output_shapes
: *
shared_namedense_2/kernel*
dtype0
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
Ф
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@dense_2/kernel
Х
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
:	Р
М
dense_2/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
в
dense_2/biasVarHandleOp*
shape:*
	container *
_class
loc:@dense_2/bias*
_output_shapes
: *
shared_namedense_2/bias*
dtype0
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
Г
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
_class
loc:@dense_2/bias*
dtype0
К
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
g
MatMul_2/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes
:	Р
О
MatMul_2MatMul	Reshape_1MatMul_2/ReadVariableOp*'
_output_shapes
:         *
transpose_b( *
transpose_a( *
T0
a
BiasAdd_2/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_2BiasAddMatMul_2BiasAdd_2/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         
K
Relu_1Relu	BiasAdd_2*'
_output_shapes
:         *
T0
f
cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

M
cond_1/switch_tIdentitycond_1/Switch:1*
_output_shapes
: *
T0

K
cond_1/switch_fIdentitycond_1/Switch*
_output_shapes
: *
T0

Q
cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

j
cond_1/dropout/rateConst^cond_1/switch_t*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
q
cond_1/dropout/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_1/dropout/Shape/SwitchSwitchRelu_1cond_1/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@Relu_1
k
cond_1/dropout/sub/xConst^cond_1/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
cond_1/dropout/subSubcond_1/dropout/sub/xcond_1/dropout/rate*
_output_shapes
: *
T0
x
!cond_1/dropout/random_uniform/minConst^cond_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_1/dropout/random_uniform/maxConst^cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
к
+cond_1/dropout/random_uniform/RandomUniformRandomUniformcond_1/dropout/Shape*
seed2 *

seed *
T0*
dtype0*'
_output_shapes
:         
П
!cond_1/dropout/random_uniform/subSub!cond_1/dropout/random_uniform/max!cond_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
к
!cond_1/dropout/random_uniform/mulMul+cond_1/dropout/random_uniform/RandomUniform!cond_1/dropout/random_uniform/sub*'
_output_shapes
:         *
T0
Ь
cond_1/dropout/random_uniformAdd!cond_1/dropout/random_uniform/mul!cond_1/dropout/random_uniform/min*'
_output_shapes
:         *
T0
~
cond_1/dropout/addAddcond_1/dropout/subcond_1/dropout/random_uniform*'
_output_shapes
:         *
T0
c
cond_1/dropout/FloorFloorcond_1/dropout/add*
T0*'
_output_shapes
:         
Ж
cond_1/dropout/truedivRealDivcond_1/dropout/Shape/Switch:1cond_1/dropout/sub*'
_output_shapes
:         *
T0
y
cond_1/dropout/mulMulcond_1/dropout/truedivcond_1/dropout/Floor*'
_output_shapes
:         *
T0
e
cond_1/IdentityIdentitycond_1/Identity/Switch*'
_output_shapes
:         *
T0
Ш
cond_1/Identity/SwitchSwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         
w
cond_1/MergeMergecond_1/Identitycond_1/dropout/mul*
T0*
N*)
_output_shapes
:         : 
г
/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:
Х
-dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *ЇЇї╛*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
Х
-dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЇЇї>*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
ы
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
*
dtype0
╓
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_3/kernel
ш
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_3/kernel*
_output_shapes

:
*
T0
┌
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:

м
dense_3/kernelVarHandleOp*
dtype0*
shape
:
*
	container *!
_class
loc:@dense_3/kernel*
_output_shapes
: *
shared_namedense_3/kernel
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Ф
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*!
_class
loc:@dense_3/kernel*
dtype0
Ф
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:
*!
_class
loc:@dense_3/kernel
М
dense_3/bias/Initializer/zerosConst*
_output_shapes
:
*
valueB
*    *
_class
loc:@dense_3/bias*
dtype0
в
dense_3/biasVarHandleOp*
dtype0*
shape:
*
	container *
_class
loc:@dense_3/bias*
_output_shapes
: *
shared_namedense_3/bias
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
Г
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
_class
loc:@dense_3/bias*
dtype0
К
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:

f
MatMul_3/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:

С
MatMul_3MatMulcond_1/MergeMatMul_3/ReadVariableOp*'
_output_shapes
:         
*
transpose_b( *
transpose_a( *
T0
a
BiasAdd_3/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_3BiasAddMatMul_3BiasAdd_3/ReadVariableOp*'
_output_shapes
:         
*
T0*
data_formatNHWC
Q
	Softmax_1Softmax	BiasAdd_3*'
_output_shapes
:         
*
T0
Ж
output_1_target_1Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
T
Const_1Const*
dtype0*
_output_shapes
:*
valueB*  А?
И
output_1_sample_weights_1PlaceholderWithDefaultConst_1*
shape:         *
dtype0*#
_output_shapes
:         
z
total_1/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@total_1*
dtype0
П
total_1VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@total_1*
_output_shapes
: *
shared_name	total_1
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
o
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
_class
loc:@total_1*
dtype0
w
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_class
loc:@total_1*
dtype0*
_output_shapes
: 
z
count_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@count_1*
dtype0*
_output_shapes
: 
П
count_1VarHandleOp*
shape: *
	container *
_class
loc:@count_1*
_output_shapes
: *
shared_name	count_1*
dtype0
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
o
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
_class
loc:@count_1*
dtype0
w
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
_class
loc:@count_1*
dtype0
u
"loss_1/output_1_loss/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Ъ
loss_1/output_1_loss/ReshapeReshapeoutput_1_target_1"loss_1/output_1_loss/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
М
loss_1/output_1_loss/CastCastloss_1/output_1_loss/Reshape*
Truncate( *

SrcT0*

DstT0	*#
_output_shapes
:         
u
$loss_1/output_1_loss/Reshape_1/shapeConst*
valueB"    
   *
dtype0*
_output_shapes
:
Ъ
loss_1/output_1_loss/Reshape_1Reshape	BiasAdd_3$loss_1/output_1_loss/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:         

Ч
>loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_1/output_1_loss/Cast*
_output_shapes
:*
T0	*
out_type0
О
\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_1/output_1_loss/Reshape_1loss_1/output_1_loss/Cast*6
_output_shapes$
":         :         
*
T0*
Tlabels0	
в
Iloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights_1*
T0*
out_type0*
_output_shapes
:
К
Hloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
ф
Hloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
Й
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Й
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
В
Eloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualGloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
М
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
╒
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentitySloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

╙
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

╞
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

ї
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*X
_classN
LJloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
С
qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Ю
xloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rankRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
в
zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : *
T0
■
kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

Й
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
З
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitykloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

М
lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
└
Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Аloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЛloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
╕
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
Ф
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЗloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchlloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
╟
Еloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
:*
valueB"      *
dtype0
╕
Еloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
value	B :*
dtype0
╥
loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЕloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЕloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
┤
Бloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╬
|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Аloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeБloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
┬
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
▀
Вloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsНloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*
T0*

Tdim0
╝
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Щ
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЙloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchlloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape
е
Оloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationВloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*<
_output_shapes*
(:         :         :*
set_operationa-b*
validate_indices(*
T0
╙
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeРloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
й
wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
б
uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЖloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
В
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranklloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Д
_classz
xvloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Е
jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergemloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N
╚
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergejloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
й
Aloss_1/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
Т
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
Я
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*,
value#B! Boutput_1_sample_weights_1:0*
dtype0*
_output_shapes
: 
С
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
т
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*o
valuefBd B^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
О
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Я
Nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergePloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

╧
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
═
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

╬
Oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
з
Lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Н
Zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tM^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
_output_shapes
: *
T0
*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Р
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
ў
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
Д
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *,
value#B! Boutput_1_sample_weights_1:0
Ў
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
╟
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*o
valuefBd B^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
є
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB B
is_scalar=*
dtype0
щ
Nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

К
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
Ж
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchIloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Д
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
Ў
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*X
_classN
LJloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
С
\loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fO^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
T0

╝
Mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMerge\loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
в
6loss_1/output_1_loss/broadcast_weights/ones_like/ShapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsN^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
╦
6loss_1/output_1_loss/broadcast_weights/ones_like/ConstConstN^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ш
0loss_1/output_1_loss/broadcast_weights/ones_likeFill6loss_1/output_1_loss/broadcast_weights/ones_like/Shape6loss_1/output_1_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0*

index_type0
и
&loss_1/output_1_loss/broadcast_weightsMuloutput_1_sample_weights_10loss_1/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
╙
loss_1/output_1_loss/MulMul\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&loss_1/output_1_loss/broadcast_weights*
T0*#
_output_shapes
:         
d
loss_1/output_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
У
loss_1/output_1_loss/SumSumloss_1/output_1_loss/Mulloss_1/output_1_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
f
loss_1/output_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
е
loss_1/output_1_loss/Sum_1Sum&loss_1/output_1_loss/broadcast_weightsloss_1/output_1_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
В
loss_1/output_1_loss/div_no_nanDivNoNanloss_1/output_1_loss/Sumloss_1/output_1_loss/Sum_1*
T0*
_output_shapes
: 
_
loss_1/output_1_loss/Const_2Const*
dtype0*
_output_shapes
: *
valueB 
Ю
loss_1/output_1_loss/MeanMeanloss_1/output_1_loss/div_no_nanloss_1/output_1_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Q
loss_1/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
[

loss_1/mulMulloss_1/mul/xloss_1/output_1_loss/Mean*
_output_shapes
: *
T0
З
metrics_1/acc/CastCastoutput_1_target_1*0
_output_shapes
:                  *
Truncate( *

SrcT0*

DstT0
В
metrics_1/acc/SqueezeSqueezemetrics_1/acc/Cast*#
_output_shapes
:         *
T0*
squeeze_dims

         
i
metrics_1/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ц
metrics_1/acc/ArgMaxArgMax	Softmax_1metrics_1/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0

metrics_1/acc/Cast_1Castmetrics_1/acc/ArgMax*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
w
metrics_1/acc/EqualEqualmetrics_1/acc/Squeezemetrics_1/acc/Cast_1*#
_output_shapes
:         *
T0
~
metrics_1/acc/Cast_2Castmetrics_1/acc/Equal*
Truncate( *

SrcT0
*

DstT0*#
_output_shapes
:         
a
metrics_1/acc/SizeSizemetrics_1/acc/Cast_2*
T0*
out_type0*
_output_shapes
: 
p
metrics_1/acc/Cast_3Castmetrics_1/acc/Size*
_output_shapes
: *
Truncate( *

SrcT0*

DstT0
]
metrics_1/acc/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Б
metrics_1/acc/SumSummetrics_1/acc/Cast_2metrics_1/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
!metrics_1/acc/AssignAddVariableOpAssignAddVariableOptotal_1metrics_1/acc/Sum*
dtype0
А
metrics_1/acc/ReadVariableOpReadVariableOptotal_1"^metrics_1/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
Е
#metrics_1/acc/AssignAddVariableOp_1AssignAddVariableOpcount_1metrics_1/acc/Cast_3^metrics_1/acc/ReadVariableOp*
dtype0
г
metrics_1/acc/ReadVariableOp_1ReadVariableOpcount_1$^metrics_1/acc/AssignAddVariableOp_1^metrics_1/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
И
'metrics_1/acc/div_no_nan/ReadVariableOpReadVariableOptotal_1^metrics_1/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
К
)metrics_1/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_1^metrics_1/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Щ
metrics_1/acc/div_no_nanDivNoNan'metrics_1/acc/div_no_nan/ReadVariableOp)metrics_1/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Г
metrics_1/acc/Squeeze_1Squeezeoutput_1_target_1*
T0*
squeeze_dims

         *#
_output_shapes
:         
k
 metrics_1/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
metrics_1/acc/ArgMax_1ArgMax	Softmax_1 metrics_1/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
Б
metrics_1/acc/Cast_4Castmetrics_1/acc/ArgMax_1*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
{
metrics_1/acc/Equal_1Equalmetrics_1/acc/Squeeze_1metrics_1/acc/Cast_4*
T0*#
_output_shapes
:         
А
metrics_1/acc/Cast_5Castmetrics_1/acc/Equal_1*#
_output_shapes
:         *
Truncate( *

SrcT0
*

DstT0
_
metrics_1/acc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Е
metrics_1/acc/MeanMeanmetrics_1/acc/Cast_5metrics_1/acc/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
А
training_2/SGD/gradients/ShapeConst*
valueB *
_class
loc:@loss_1/mul*
dtype0*
_output_shapes
: 
Ж
"training_2/SGD/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
_class
loc:@loss_1/mul*
dtype0
╗
training_2/SGD/gradients/FillFilltraining_2/SGD/gradients/Shape"training_2/SGD/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0*
_class
loc:@loss_1/mul
н
,training_2/SGD/gradients/loss_1/mul_grad/MulMultraining_2/SGD/gradients/Fillloss_1/output_1_loss/Mean*
_output_shapes
: *
T0*
_class
loc:@loss_1/mul
в
.training_2/SGD/gradients/loss_1/mul_grad/Mul_1Multraining_2/SGD/gradients/Fillloss_1/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss_1/mul
╢
Etraining_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB *,
_class"
 loc:@loss_1/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ю
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/ReshapeReshape.training_2/SGD/gradients/loss_1/mul_grad/Mul_1Etraining_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0*,
_class"
 loc:@loss_1/output_1_loss/Mean
о
=training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB *,
_class"
 loc:@loss_1/output_1_loss/Mean
е
<training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/TileTile?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape=training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const*,
_class"
 loc:@loss_1/output_1_loss/Mean*
_output_shapes
: *
T0*

Tmultiples0
▓
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const_1Const*
valueB
 *  А?*,
_class"
 loc:@loss_1/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ш
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivRealDiv<training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Tile?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const_1*,
_class"
 loc:@loss_1/output_1_loss/Mean*
_output_shapes
: *
T0
║
Ctraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB *2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan
╝
Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1Const*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: *
valueB 
щ
Straining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ShapeEtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*2
_output_shapes 
:         :         
Ж
Htraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivloss_1/output_1_loss/Sum_1*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
┘
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/SumSumHtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nanStraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *

Tidx0*
	keep_dims( 
╗
Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ReshapeReshapeAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/SumCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape*
_output_shapes
: *
T0*
Tshape0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan
╖
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/NegNegloss_1/output_1_loss/Sum*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
К
Jtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNanAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Negloss_1/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: 
У
Jtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanJtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_1loss_1/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: 
к
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/mulMul?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivJtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_2*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╓
Ctraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Sum_1SumAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/mulUtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
┴
Gtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Reshape_1ReshapeCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Sum_1Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1*
Tshape0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╗
Dtraining_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:*+
_class!
loc:@loss_1/output_1_loss/Sum
╢
>training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/ReshapeReshapeEtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ReshapeDtraining_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*+
_class!
loc:@loss_1/output_1_loss/Sum
┴
<training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/ShapeShapeloss_1/output_1_loss/Mul*+
_class!
loc:@loss_1/output_1_loss/Sum*
_output_shapes
:*
T0*
out_type0
о
;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/TileTile>training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape<training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Shape*#
_output_shapes
:         *
T0*

Tmultiples0*+
_class!
loc:@loss_1/output_1_loss/Sum
Е
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ShapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0*+
_class!
loc:@loss_1/output_1_loss/Mul
╤
>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1Shape&loss_1/output_1_loss/broadcast_weights*
out_type0*+
_class!
loc:@loss_1/output_1_loss/Mul*
_output_shapes
:*
T0
═
Ltraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1*+
_class!
loc:@loss_1/output_1_loss/Mul*2
_output_shapes 
:         :         *
T0
Б
:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/MulMul;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Tile&loss_1/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0*+
_class!
loc:@loss_1/output_1_loss/Mul
╕
:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/SumSum:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/MulLtraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgs*+
_class!
loc:@loss_1/output_1_loss/Mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
м
>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ReshapeReshape:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape*
Tshape0*+
_class!
loc:@loss_1/output_1_loss/Mul*#
_output_shapes
:         *
T0
╣
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Mul_1Mul\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Tile*+
_class!
loc:@loss_1/output_1_loss/Mul*#
_output_shapes
:         *
T0
╛
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum_1Sum<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Mul_1Ntraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss_1/output_1_loss/Mul
▓
@training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Reshape_1Reshape<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum_1>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1*
Tshape0*+
_class!
loc:@loss_1/output_1_loss/Mul*#
_output_shapes
:         *
T0
│
#training_2/SGD/gradients/zeros_like	ZerosLike^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

╪
Кtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╞
Йtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0*
_output_shapes
: 
П
Еtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ReshapeЙtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
╜
~training_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulЕtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsКtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:         
*
T0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╛
Btraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_3*1
_class'
%#loc:@loss_1/output_1_loss/Reshape_1*
_output_shapes
:*
T0*
out_type0
Ж
Dtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ReshapeReshape~training_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulBtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Shape*
T0*
Tshape0*1
_class'
%#loc:@loss_1/output_1_loss/Reshape_1*'
_output_shapes
:         

т
3training_2/SGD/gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@BiasAdd_3*
_output_shapes
:
*
T0*
data_formatNHWC
Л
-training_2/SGD/gradients/MatMul_3_grad/MatMulMatMulDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ReshapeMatMul_3/ReadVariableOp*'
_output_shapes
:         *
transpose_b(*
T0*
transpose_a( *
_class
loc:@MatMul_3
∙
/training_2/SGD/gradients/MatMul_3_grad/MatMul_1MatMulcond_1/MergeDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@MatMul_3*
_output_shapes

:
*
transpose_b( *
T0*
transpose_a(
▀
4training_2/SGD/gradients/cond_1/Merge_grad/cond_gradSwitch-training_2/SGD/gradients/MatMul_3_grad/MatMulcond_1/pred_id*
_class
loc:@MatMul_3*:
_output_shapes(
&:         :         *
T0
│
6training_2/SGD/gradients/cond_1/dropout/mul_grad/ShapeShapecond_1/dropout/truediv*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:*
T0*
out_type0
│
8training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1Shapecond_1/dropout/Floor*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:*
T0*
out_type0
╡
Ftraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape8training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_1/dropout/mul*2
_output_shapes 
:         :         
т
4training_2/SGD/gradients/cond_1/dropout/mul_grad/MulMul6training_2/SGD/gradients/cond_1/Merge_grad/cond_grad:1cond_1/dropout/Floor*%
_class
loc:@cond_1/dropout/mul*'
_output_shapes
:         *
T0
а
4training_2/SGD/gradients/cond_1/dropout/mul_grad/SumSum4training_2/SGD/gradients/cond_1/dropout/mul_grad/MulFtraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ш
8training_2/SGD/gradients/cond_1/dropout/mul_grad/ReshapeReshape4training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum6training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape*
Tshape0*%
_class
loc:@cond_1/dropout/mul*'
_output_shapes
:         *
T0
ц
6training_2/SGD/gradients/cond_1/dropout/mul_grad/Mul_1Mulcond_1/dropout/truediv6training_2/SGD/gradients/cond_1/Merge_grad/cond_grad:1*'
_output_shapes
:         *
T0*%
_class
loc:@cond_1/dropout/mul
ж
6training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum_1Sum6training_2/SGD/gradients/cond_1/dropout/mul_grad/Mul_1Htraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@cond_1/dropout/mul
Ю
:training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshape_1Reshape6training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum_18training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0*%
_class
loc:@cond_1/dropout/mul
б
training_2/SGD/gradients/SwitchSwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         
Э
!training_2/SGD/gradients/IdentityIdentity!training_2/SGD/gradients/Switch:1*
_class
loc:@Relu_1*'
_output_shapes
:         *
T0
Ь
 training_2/SGD/gradients/Shape_1Shape!training_2/SGD/gradients/Switch:1*
_class
loc:@Relu_1*
_output_shapes
:*
T0*
out_type0
и
$training_2/SGD/gradients/zeros/ConstConst"^training_2/SGD/gradients/Identity*
_output_shapes
: *
valueB
 *    *
_class
loc:@Relu_1*
dtype0
═
training_2/SGD/gradients/zerosFill training_2/SGD/gradients/Shape_1$training_2/SGD/gradients/zeros/Const*
T0*

index_type0*
_class
loc:@Relu_1*'
_output_shapes
:         
ї
>training_2/SGD/gradients/cond_1/Identity/Switch_grad/cond_gradMerge4training_2/SGD/gradients/cond_1/Merge_grad/cond_gradtraining_2/SGD/gradients/zeros*
_class
loc:@Relu_1*)
_output_shapes
:         : *
N*
T0
┬
:training_2/SGD/gradients/cond_1/dropout/truediv_grad/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*
out_type0*)
_class
loc:@cond_1/dropout/truediv*
_output_shapes
:
к
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *)
_class
loc:@cond_1/dropout/truediv
┼
Jtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1*
T0*)
_class
loc:@cond_1/dropout/truediv*2
_output_shapes 
:         :         
Є
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDivRealDiv8training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshapecond_1/dropout/sub*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         *
T0
┤
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/SumSum<training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDivJtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@cond_1/dropout/truediv
и
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/ReshapeReshape8training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape*
T0*
Tshape0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
╗
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/NegNegcond_1/dropout/Shape/Switch:1*'
_output_shapes
:         *
T0*)
_class
loc:@cond_1/dropout/truediv
Ї
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_1RealDiv8training_2/SGD/gradients/cond_1/dropout/truediv_grad/Negcond_1/dropout/sub*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         *
T0
·
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_2RealDiv>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_1cond_1/dropout/sub*'
_output_shapes
:         *
T0*)
_class
loc:@cond_1/dropout/truediv
Ц
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/mulMul8training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshape>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_2*
T0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
┤
:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum_1Sum8training_2/SGD/gradients/cond_1/dropout/truediv_grad/mulLtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@cond_1/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
Э
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/Reshape_1Reshape:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum_1<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1*
Tshape0*)
_class
loc:@cond_1/dropout/truediv*
_output_shapes
: *
T0
г
!training_2/SGD/gradients/Switch_1SwitchRelu_1cond_1/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@Relu_1
Я
#training_2/SGD/gradients/Identity_1Identity!training_2/SGD/gradients/Switch_1*
_class
loc:@Relu_1*'
_output_shapes
:         *
T0
Ь
 training_2/SGD/gradients/Shape_2Shape!training_2/SGD/gradients/Switch_1*
_class
loc:@Relu_1*
_output_shapes
:*
T0*
out_type0
м
&training_2/SGD/gradients/zeros_1/ConstConst$^training_2/SGD/gradients/Identity_1*
valueB
 *    *
_class
loc:@Relu_1*
dtype0*
_output_shapes
: 
╤
 training_2/SGD/gradients/zeros_1Fill training_2/SGD/gradients/Shape_2&training_2/SGD/gradients/zeros_1/Const*

index_type0*
_class
loc:@Relu_1*'
_output_shapes
:         *
T0
Д
Ctraining_2/SGD/gradients/cond_1/dropout/Shape/Switch_grad/cond_gradMerge training_2/SGD/gradients/zeros_1<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Reshape*
N*
T0*
_class
loc:@Relu_1*)
_output_shapes
:         : 
А
training_2/SGD/gradients/AddNAddN>training_2/SGD/gradients/cond_1/Identity/Switch_grad/cond_gradCtraining_2/SGD/gradients/cond_1/dropout/Shape/Switch_grad/cond_grad*
_class
loc:@Relu_1*'
_output_shapes
:         *
N*
T0
н
-training_2/SGD/gradients/Relu_1_grad/ReluGradReluGradtraining_2/SGD/gradients/AddNRelu_1*
T0*
_class
loc:@Relu_1*'
_output_shapes
:         
╦
3training_2/SGD/gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad-training_2/SGD/gradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_class
loc:@BiasAdd_2*
_output_shapes
:
ї
-training_2/SGD/gradients/MatMul_2_grad/MatMulMatMul-training_2/SGD/gradients/Relu_1_grad/ReluGradMatMul_2/ReadVariableOp*
T0*
transpose_a( *
_class
loc:@MatMul_2*(
_output_shapes
:         Р*
transpose_b(
р
/training_2/SGD/gradients/MatMul_2_grad/MatMul_1MatMul	Reshape_1-training_2/SGD/gradients/Relu_1_grad/ReluGrad*
_class
loc:@MatMul_2*
_output_shapes
:	Р*
transpose_b( *
T0*
transpose_a(
V
training_2/SGD/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R
l
"training_2/SGD/AssignAddVariableOpAssignAddVariableOpSGD/iterationstraining_2/SGD/Const*
dtype0	
Й
training_2/SGD/ReadVariableOpReadVariableOpSGD/iterations#^training_2/SGD/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
u
$training_2/SGD/zeros/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
_
training_2/SGD/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ъ
training_2/SGD/zerosFill$training_2/SGD/zeros/shape_as_tensortraining_2/SGD/zeros/Const*
T0*

index_type0*
_output_shapes
:	Р
╚
training_2/SGD/VariableVarHandleOp*
shape:	Р*
	container **
_class 
loc:@training_2/SGD/Variable*
_output_shapes
: *(
shared_nametraining_2/SGD/Variable*
dtype0

8training_2/SGD/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable*
_output_shapes
: 
Ъ
training_2/SGD/Variable/AssignAssignVariableOptraining_2/SGD/Variabletraining_2/SGD/zeros**
_class 
loc:@training_2/SGD/Variable*
dtype0
░
+training_2/SGD/Variable/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable*
_output_shapes
:	Р**
_class 
loc:@training_2/SGD/Variable*
dtype0
c
training_2/SGD/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
╔
training_2/SGD/Variable_1VarHandleOp*
dtype0*
shape:*
	container *,
_class"
 loc:@training_2/SGD/Variable_1*
_output_shapes
: **
shared_nametraining_2/SGD/Variable_1
Г
:training_2/SGD/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_1*
_output_shapes
: 
в
 training_2/SGD/Variable_1/AssignAssignVariableOptraining_2/SGD/Variable_1training_2/SGD/zeros_1*,
_class"
 loc:@training_2/SGD/Variable_1*
dtype0
▒
-training_2/SGD/Variable_1/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_1*,
_class"
 loc:@training_2/SGD/Variable_1*
dtype0*
_output_shapes
:
k
training_2/SGD/zeros_2Const*
dtype0*
_output_shapes

:
*
valueB
*    
═
training_2/SGD/Variable_2VarHandleOp*
shape
:
*
	container *,
_class"
 loc:@training_2/SGD/Variable_2*
_output_shapes
: **
shared_nametraining_2/SGD/Variable_2*
dtype0
Г
:training_2/SGD/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_2*
_output_shapes
: 
в
 training_2/SGD/Variable_2/AssignAssignVariableOptraining_2/SGD/Variable_2training_2/SGD/zeros_2*
dtype0*,
_class"
 loc:@training_2/SGD/Variable_2
╡
-training_2/SGD/Variable_2/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_2*,
_class"
 loc:@training_2/SGD/Variable_2*
dtype0*
_output_shapes

:

c
training_2/SGD/zeros_3Const*
_output_shapes
:
*
valueB
*    *
dtype0
╔
training_2/SGD/Variable_3VarHandleOp*
shape:
*
	container *,
_class"
 loc:@training_2/SGD/Variable_3*
_output_shapes
: **
shared_nametraining_2/SGD/Variable_3*
dtype0
Г
:training_2/SGD/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_3*
_output_shapes
: 
в
 training_2/SGD/Variable_3/AssignAssignVariableOptraining_2/SGD/Variable_3training_2/SGD/zeros_3*,
_class"
 loc:@training_2/SGD/Variable_3*
dtype0
▒
-training_2/SGD/Variable_3/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_3*,
_class"
 loc:@training_2/SGD/Variable_3*
dtype0*
_output_shapes
:

d
training_2/SGD/ReadVariableOp_1ReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
z
!training_2/SGD/mul/ReadVariableOpReadVariableOptraining_2/SGD/Variable*
dtype0*
_output_shapes
:	Р
З
training_2/SGD/mulMultraining_2/SGD/ReadVariableOp_1!training_2/SGD/mul/ReadVariableOp*
_output_shapes
:	Р*
T0
^
training_2/SGD/ReadVariableOp_2ReadVariableOpSGD/lr*
_output_shapes
: *
dtype0
Ч
training_2/SGD/mul_1Multraining_2/SGD/ReadVariableOp_2/training_2/SGD/gradients/MatMul_2_grad/MatMul_1*
_output_shapes
:	Р*
T0
m
training_2/SGD/subSubtraining_2/SGD/multraining_2/SGD/mul_1*
T0*
_output_shapes
:	Р
m
training_2/SGD/AssignVariableOpAssignVariableOptraining_2/SGD/Variabletraining_2/SGD/sub*
dtype0
Ъ
training_2/SGD/ReadVariableOp_3ReadVariableOptraining_2/SGD/Variable ^training_2/SGD/AssignVariableOp*
dtype0*
_output_shapes
:	Р
o
training_2/SGD/ReadVariableOp_4ReadVariableOpdense_2/kernel*
dtype0*
_output_shapes
:	Р
x
training_2/SGD/addAddtraining_2/SGD/ReadVariableOp_4training_2/SGD/sub*
T0*
_output_shapes
:	Р
f
!training_2/SGD/AssignVariableOp_1AssignVariableOpdense_2/kerneltraining_2/SGD/add*
dtype0
У
training_2/SGD/ReadVariableOp_5ReadVariableOpdense_2/kernel"^training_2/SGD/AssignVariableOp_1*
_output_shapes
:	Р*
dtype0
d
training_2/SGD/ReadVariableOp_6ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
y
#training_2/SGD/mul_2/ReadVariableOpReadVariableOptraining_2/SGD/Variable_1*
dtype0*
_output_shapes
:
Ж
training_2/SGD/mul_2Multraining_2/SGD/ReadVariableOp_6#training_2/SGD/mul_2/ReadVariableOp*
T0*
_output_shapes
:
^
training_2/SGD/ReadVariableOp_7ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ц
training_2/SGD/mul_3Multraining_2/SGD/ReadVariableOp_73training_2/SGD/gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training_2/SGD/sub_1Subtraining_2/SGD/mul_2training_2/SGD/mul_3*
T0*
_output_shapes
:
s
!training_2/SGD/AssignVariableOp_2AssignVariableOptraining_2/SGD/Variable_1training_2/SGD/sub_1*
dtype0
Щ
training_2/SGD/ReadVariableOp_8ReadVariableOptraining_2/SGD/Variable_1"^training_2/SGD/AssignVariableOp_2*
dtype0*
_output_shapes
:
h
training_2/SGD/ReadVariableOp_9ReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
w
training_2/SGD/add_1Addtraining_2/SGD/ReadVariableOp_9training_2/SGD/sub_1*
T0*
_output_shapes
:
f
!training_2/SGD/AssignVariableOp_3AssignVariableOpdense_2/biastraining_2/SGD/add_1*
dtype0
Н
 training_2/SGD/ReadVariableOp_10ReadVariableOpdense_2/bias"^training_2/SGD/AssignVariableOp_3*
dtype0*
_output_shapes
:
e
 training_2/SGD/ReadVariableOp_11ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
}
#training_2/SGD/mul_4/ReadVariableOpReadVariableOptraining_2/SGD/Variable_2*
dtype0*
_output_shapes

:

Л
training_2/SGD/mul_4Mul training_2/SGD/ReadVariableOp_11#training_2/SGD/mul_4/ReadVariableOp*
_output_shapes

:
*
T0
_
 training_2/SGD/ReadVariableOp_12ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ч
training_2/SGD/mul_5Mul training_2/SGD/ReadVariableOp_12/training_2/SGD/gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:
*
T0
p
training_2/SGD/sub_2Subtraining_2/SGD/mul_4training_2/SGD/mul_5*
_output_shapes

:
*
T0
s
!training_2/SGD/AssignVariableOp_4AssignVariableOptraining_2/SGD/Variable_2training_2/SGD/sub_2*
dtype0
Ю
 training_2/SGD/ReadVariableOp_13ReadVariableOptraining_2/SGD/Variable_2"^training_2/SGD/AssignVariableOp_4*
dtype0*
_output_shapes

:

o
 training_2/SGD/ReadVariableOp_14ReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:

|
training_2/SGD/add_2Add training_2/SGD/ReadVariableOp_14training_2/SGD/sub_2*
T0*
_output_shapes

:

h
!training_2/SGD/AssignVariableOp_5AssignVariableOpdense_3/kerneltraining_2/SGD/add_2*
dtype0
У
 training_2/SGD/ReadVariableOp_15ReadVariableOpdense_3/kernel"^training_2/SGD/AssignVariableOp_5*
dtype0*
_output_shapes

:

e
 training_2/SGD/ReadVariableOp_16ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
y
#training_2/SGD/mul_6/ReadVariableOpReadVariableOptraining_2/SGD/Variable_3*
dtype0*
_output_shapes
:

З
training_2/SGD/mul_6Mul training_2/SGD/ReadVariableOp_16#training_2/SGD/mul_6/ReadVariableOp*
_output_shapes
:
*
T0
_
 training_2/SGD/ReadVariableOp_17ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ч
training_2/SGD/mul_7Mul training_2/SGD/ReadVariableOp_173training_2/SGD/gradients/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:
*
T0
l
training_2/SGD/sub_3Subtraining_2/SGD/mul_6training_2/SGD/mul_7*
_output_shapes
:
*
T0
s
!training_2/SGD/AssignVariableOp_6AssignVariableOptraining_2/SGD/Variable_3training_2/SGD/sub_3*
dtype0
Ъ
 training_2/SGD/ReadVariableOp_18ReadVariableOptraining_2/SGD/Variable_3"^training_2/SGD/AssignVariableOp_6*
dtype0*
_output_shapes
:

i
 training_2/SGD/ReadVariableOp_19ReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:

x
training_2/SGD/add_3Add training_2/SGD/ReadVariableOp_19training_2/SGD/sub_3*
_output_shapes
:
*
T0
f
!training_2/SGD/AssignVariableOp_7AssignVariableOpdense_3/biastraining_2/SGD/add_3*
dtype0
Н
 training_2/SGD/ReadVariableOp_20ReadVariableOpdense_3/bias"^training_2/SGD/AssignVariableOp_7*
dtype0*
_output_shapes
:

·
training_3/group_depsNoOp^loss_1/mul^metrics_1/acc/div_no_nan^training_2/SGD/ReadVariableOp!^training_2/SGD/ReadVariableOp_10!^training_2/SGD/ReadVariableOp_13!^training_2/SGD/ReadVariableOp_15!^training_2/SGD/ReadVariableOp_18!^training_2/SGD/ReadVariableOp_20 ^training_2/SGD/ReadVariableOp_3 ^training_2/SGD/ReadVariableOp_5 ^training_2/SGD/ReadVariableOp_8
\
VarIsInitializedOp_23VarIsInitializedOptraining_2/SGD/Variable*
_output_shapes
: 
^
VarIsInitializedOp_24VarIsInitializedOptraining_2/SGD/Variable_1*
_output_shapes
: 
^
VarIsInitializedOp_25VarIsInitializedOptraining_2/SGD/Variable_2*
_output_shapes
: 
S
VarIsInitializedOp_26VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_27VarIsInitializedOpdense_2/bias*
_output_shapes
: 
Q
VarIsInitializedOp_28VarIsInitializedOpdense_3/bias*
_output_shapes
: 
L
VarIsInitializedOp_29VarIsInitializedOptotal_1*
_output_shapes
: 
S
VarIsInitializedOp_30VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_31VarIsInitializedOpSGD/momentum*
_output_shapes
: 
K
VarIsInitializedOp_32VarIsInitializedOpSGD/lr*
_output_shapes
: 
S
VarIsInitializedOp_33VarIsInitializedOpSGD/iterations*
_output_shapes
: 
^
VarIsInitializedOp_34VarIsInitializedOptraining_2/SGD/Variable_3*
_output_shapes
: 
L
VarIsInitializedOp_35VarIsInitializedOpcount_1*
_output_shapes
: 
N
VarIsInitializedOp_36VarIsInitializedOp	SGD/decay*
_output_shapes
: 
ч
init_1NoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^count_1/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^total_1/Assign^training_2/SGD/Variable/Assign!^training_2/SGD/Variable_1/Assign!^training_2/SGD/Variable_2/Assign!^training_2/SGD/Variable_3/Assign
N
Placeholder_2Placeholder*
shape: *
dtype0*
_output_shapes
: 
K
AssignVariableOp_2AssignVariableOptotal_1Placeholder_2*
dtype0
e
ReadVariableOp_2ReadVariableOptotal_1^AssignVariableOp_2*
dtype0*
_output_shapes
: 
N
Placeholder_3Placeholder*
shape: *
dtype0*
_output_shapes
: 
K
AssignVariableOp_3AssignVariableOpcount_1Placeholder_3*
dtype0
e
ReadVariableOp_3ReadVariableOpcount_1^AssignVariableOp_3*
dtype0*
_output_shapes
: 
G
evaluation_1/group_depsNoOp^loss_1/mul^metrics_1/acc/div_no_nan
У
+Adam_1/iterations/Initializer/initial_valueConst*$
_class
loc:@Adam_1/iterations*
dtype0	*
_output_shapes
: *
value	B	 R 
н
Adam_1/iterationsVarHandleOp*
dtype0	*
shape: *
	container *$
_class
loc:@Adam_1/iterations*
_output_shapes
: *"
shared_nameAdam_1/iterations
s
2Adam_1/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/iterations*
_output_shapes
: 
Я
Adam_1/iterations/AssignAssignVariableOpAdam_1/iterations+Adam_1/iterations/Initializer/initial_value*$
_class
loc:@Adam_1/iterations*
dtype0	
Х
%Adam_1/iterations/Read/ReadVariableOpReadVariableOpAdam_1/iterations*
_output_shapes
: *$
_class
loc:@Adam_1/iterations*
dtype0	
Ж
#Adam_1/lr/Initializer/initial_valueConst*
valueB
 *oГ:*
_class
loc:@Adam_1/lr*
dtype0*
_output_shapes
: 
Х
	Adam_1/lrVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam_1/lr*
_output_shapes
: *
shared_name	Adam_1/lr
c
*Adam_1/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOp	Adam_1/lr*
_output_shapes
: 

Adam_1/lr/AssignAssignVariableOp	Adam_1/lr#Adam_1/lr/Initializer/initial_value*
_class
loc:@Adam_1/lr*
dtype0
}
Adam_1/lr/Read/ReadVariableOpReadVariableOp	Adam_1/lr*
_class
loc:@Adam_1/lr*
dtype0*
_output_shapes
: 
О
'Adam_1/beta_1/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?* 
_class
loc:@Adam_1/beta_1
б
Adam_1/beta_1VarHandleOp*
shape: *
	container * 
_class
loc:@Adam_1/beta_1*
_output_shapes
: *
shared_nameAdam_1/beta_1*
dtype0
k
.Adam_1/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/beta_1*
_output_shapes
: 
П
Adam_1/beta_1/AssignAssignVariableOpAdam_1/beta_1'Adam_1/beta_1/Initializer/initial_value* 
_class
loc:@Adam_1/beta_1*
dtype0
Й
!Adam_1/beta_1/Read/ReadVariableOpReadVariableOpAdam_1/beta_1* 
_class
loc:@Adam_1/beta_1*
dtype0*
_output_shapes
: 
О
'Adam_1/beta_2/Initializer/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w╛?* 
_class
loc:@Adam_1/beta_2
б
Adam_1/beta_2VarHandleOp*
dtype0*
shape: *
	container * 
_class
loc:@Adam_1/beta_2*
_output_shapes
: *
shared_nameAdam_1/beta_2
k
.Adam_1/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/beta_2*
_output_shapes
: 
П
Adam_1/beta_2/AssignAssignVariableOpAdam_1/beta_2'Adam_1/beta_2/Initializer/initial_value* 
_class
loc:@Adam_1/beta_2*
dtype0
Й
!Adam_1/beta_2/Read/ReadVariableOpReadVariableOpAdam_1/beta_2* 
_class
loc:@Adam_1/beta_2*
dtype0*
_output_shapes
: 
М
&Adam_1/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@Adam_1/decay*
dtype0*
_output_shapes
: 
Ю
Adam_1/decayVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam_1/decay*
_output_shapes
: *
shared_nameAdam_1/decay
i
-Adam_1/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/decay*
_output_shapes
: 
Л
Adam_1/decay/AssignAssignVariableOpAdam_1/decay&Adam_1/decay/Initializer/initial_value*
_class
loc:@Adam_1/decay*
dtype0
Ж
 Adam_1/decay/Read/ReadVariableOpReadVariableOpAdam_1/decay*
_class
loc:@Adam_1/decay*
dtype0*
_output_shapes
: 
t
	input_1_2Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
P
Shape_2Shape	input_1_2*
_output_shapes
:*
T0*
out_type0
_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
strided_slice_2StridedSliceShape_2strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
Index0*
new_axis_mask *
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
ellipsis_mask *
end_mask 
\
Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
         
u
Reshape_2/shapePackstrided_slice_2Reshape_2/shape/1*
T0*
N*
_output_shapes
:*

axis 
q
	Reshape_2Reshape	input_1_2Reshape_2/shape*
T0*
Tshape0*(
_output_shapes
:         Р
г
/dense_4/kernel/Initializer/random_uniform/shapeConst*
valueB"     *!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
:
Х
-dense_4/kernel/Initializer/random_uniform/minConst*
valueB
 *м\▒╜*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
Х
-dense_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *м\▒=*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
ь
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
:	Р*
dtype0
╓
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
_output_shapes
: *
T0
щ
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
:	Р
█
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
:	Р*
T0*!
_class
loc:@dense_4/kernel
н
dense_4/kernelVarHandleOp*
dtype0*
shape:	Р*
	container *!
_class
loc:@dense_4/kernel*
_output_shapes
: *
shared_namedense_4/kernel
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Ф
dense_4/kernel/AssignAssignVariableOpdense_4/kernel)dense_4/kernel/Initializer/random_uniform*!
_class
loc:@dense_4/kernel*
dtype0
Х
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
:	Р
М
dense_4/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@dense_4/bias*
dtype0
в
dense_4/biasVarHandleOp*
_output_shapes
: *
shared_namedense_4/bias*
dtype0*
shape:*
	container *
_class
loc:@dense_4/bias
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
Г
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/bias/Initializer/zeros*
_class
loc:@dense_4/bias*
dtype0
К
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
:
g
MatMul_4/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0*
_output_shapes
:	Р
О
MatMul_4MatMul	Reshape_2MatMul_4/ReadVariableOp*
transpose_a( *
T0*'
_output_shapes
:         *
transpose_b( 
a
BiasAdd_4/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_4BiasAddMatMul_4BiasAdd_4/ReadVariableOp*'
_output_shapes
:         *
T0*
data_formatNHWC
K
Relu_2Relu	BiasAdd_4*'
_output_shapes
:         *
T0
f
cond_2/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

M
cond_2/switch_tIdentitycond_2/Switch:1*
_output_shapes
: *
T0

K
cond_2/switch_fIdentitycond_2/Switch*
_output_shapes
: *
T0

Q
cond_2/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
j
cond_2/dropout/rateConst^cond_2/switch_t*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
q
cond_2/dropout/ShapeShapecond_2/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_2/dropout/Shape/SwitchSwitchRelu_2cond_2/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@Relu_2
k
cond_2/dropout/sub/xConst^cond_2/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
cond_2/dropout/subSubcond_2/dropout/sub/xcond_2/dropout/rate*
_output_shapes
: *
T0
x
!cond_2/dropout/random_uniform/minConst^cond_2/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_2/dropout/random_uniform/maxConst^cond_2/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
к
+cond_2/dropout/random_uniform/RandomUniformRandomUniformcond_2/dropout/Shape*
dtype0*'
_output_shapes
:         *
seed2 *

seed *
T0
П
!cond_2/dropout/random_uniform/subSub!cond_2/dropout/random_uniform/max!cond_2/dropout/random_uniform/min*
_output_shapes
: *
T0
к
!cond_2/dropout/random_uniform/mulMul+cond_2/dropout/random_uniform/RandomUniform!cond_2/dropout/random_uniform/sub*
T0*'
_output_shapes
:         
Ь
cond_2/dropout/random_uniformAdd!cond_2/dropout/random_uniform/mul!cond_2/dropout/random_uniform/min*
T0*'
_output_shapes
:         
~
cond_2/dropout/addAddcond_2/dropout/subcond_2/dropout/random_uniform*'
_output_shapes
:         *
T0
c
cond_2/dropout/FloorFloorcond_2/dropout/add*
T0*'
_output_shapes
:         
Ж
cond_2/dropout/truedivRealDivcond_2/dropout/Shape/Switch:1cond_2/dropout/sub*
T0*'
_output_shapes
:         
y
cond_2/dropout/mulMulcond_2/dropout/truedivcond_2/dropout/Floor*
T0*'
_output_shapes
:         
e
cond_2/IdentityIdentitycond_2/Identity/Switch*'
_output_shapes
:         *
T0
Ш
cond_2/Identity/SwitchSwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         
w
cond_2/MergeMergecond_2/Identitycond_2/dropout/mul*)
_output_shapes
:         : *
T0*
N
г
/dense_5/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
:
Х
-dense_5/kernel/Initializer/random_uniform/minConst*
valueB
 *ЇЇї╛*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 
Х
-dense_5/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ЇЇї>*!
_class
loc:@dense_5/kernel*
dtype0
ы
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
_output_shapes

:
*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_5/kernel
╓
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 
ш
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:

┌
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
_output_shapes

:
*
T0
м
dense_5/kernelVarHandleOp*
dtype0*
shape
:
*
	container *!
_class
loc:@dense_5/kernel*
_output_shapes
: *
shared_namedense_5/kernel
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
Ф
dense_5/kernel/AssignAssignVariableOpdense_5/kernel)dense_5/kernel/Initializer/random_uniform*!
_class
loc:@dense_5/kernel*
dtype0
Ф
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:

М
dense_5/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:

в
dense_5/biasVarHandleOp*
dtype0*
shape:
*
	container *
_class
loc:@dense_5/bias*
_output_shapes
: *
shared_namedense_5/bias
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
Г
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/bias/Initializer/zeros*
_class
loc:@dense_5/bias*
dtype0
К
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:

f
MatMul_5/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes

:

С
MatMul_5MatMulcond_2/MergeMatMul_5/ReadVariableOp*
transpose_a( *
T0*'
_output_shapes
:         
*
transpose_b( 
a
BiasAdd_5/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_5BiasAddMatMul_5BiasAdd_5/ReadVariableOp*'
_output_shapes
:         
*
T0*
data_formatNHWC
Q
	Softmax_2Softmax	BiasAdd_5*'
_output_shapes
:         
*
T0
Ж
output_1_target_2Placeholder*0
_output_shapes
:                  *%
shape:                  *
dtype0
T
Const_2Const*
_output_shapes
:*
valueB*  А?*
dtype0
И
output_1_sample_weights_2PlaceholderWithDefaultConst_2*
shape:         *
dtype0*#
_output_shapes
:         
z
total_2/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@total_2
П
total_2VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@total_2*
_output_shapes
: *
shared_name	total_2
_
(total_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_2*
_output_shapes
: 
o
total_2/AssignAssignVariableOptotal_2total_2/Initializer/zeros*
_class
loc:@total_2*
dtype0
w
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_class
loc:@total_2*
dtype0*
_output_shapes
: 
z
count_2/Initializer/zerosConst*
valueB
 *    *
_class
loc:@count_2*
dtype0*
_output_shapes
: 
П
count_2VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@count_2*
_output_shapes
: *
shared_name	count_2
_
(count_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_2*
_output_shapes
: 
o
count_2/AssignAssignVariableOpcount_2count_2/Initializer/zeros*
dtype0*
_class
loc:@count_2
w
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_class
loc:@count_2*
dtype0*
_output_shapes
: 
u
"loss_2/output_1_loss/Reshape/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ъ
loss_2/output_1_loss/ReshapeReshapeoutput_1_target_2"loss_2/output_1_loss/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0
М
loss_2/output_1_loss/CastCastloss_2/output_1_loss/Reshape*
Truncate( *

SrcT0*

DstT0	*#
_output_shapes
:         
u
$loss_2/output_1_loss/Reshape_1/shapeConst*
valueB"    
   *
dtype0*
_output_shapes
:
Ъ
loss_2/output_1_loss/Reshape_1Reshape	BiasAdd_5$loss_2/output_1_loss/Reshape_1/shape*'
_output_shapes
:         
*
T0*
Tshape0
Ч
>loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_2/output_1_loss/Cast*
_output_shapes
:*
T0	*
out_type0
О
\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_2/output_1_loss/Reshape_1loss_2/output_1_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":         :         

в
Iloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights_2*
T0*
out_type0*
_output_shapes
:
К
Hloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
ф
Hloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
Й
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Й
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
В
Eloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualGloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
М
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
╒
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentitySloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

╙
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╞
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
ї
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*X
_classN
LJloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
С
qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
Ю
xloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rankRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*Z
_classP
NLloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : *
T0
в
zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : *
T0
■
kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

Й
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
З
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitykloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

М
lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

└
Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
valueB :
         *
dtype0
┘
Аloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЛloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
_output_shapes

:*
T0
╕
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
Ф
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЗloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchlloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
╟
Еloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╕
Еloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╥
loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЕloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЕloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*
_output_shapes

:
┤
Бloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╬
|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Аloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeБloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
_output_shapes

:*

Tidx0*
T0*
N
┬
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
▀
Вloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsНloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*
T0*

Tdim0
╝
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Щ
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЙloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchlloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
е
Оloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationВloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
validate_indices(*
T0*<
_output_shapes*
(:         :         :*
set_operationa-b
╙
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeРloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
й
wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
б
uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЖloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
В
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranklloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Д
_classz
xvloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Е
jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergemloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
╚
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergejloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
T0
*
N
й
Aloss_2/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Т
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Я
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*,
value#B! Boutput_1_sample_weights_2:0*
dtype0*
_output_shapes
: 
С
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
т
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*o
valuefBd B^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
О
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Я
Nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergePloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
╧
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
═
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

╬
Oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
з
Lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Н
Zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tM^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: *
T0

Р
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.*
dtype0
ў
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Д
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*,
value#B! Boutput_1_sample_weights_2:0*
dtype0*
_output_shapes
: 
Ў
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
╟
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*o
valuefBd B^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
є
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
щ
Nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
	summarize*
T
2	

К
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
Ж
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchIloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id* 
_output_shapes
::*
T0*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape
Д
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
Ў
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*X
_classN
LJloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

С
\loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fO^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╝
Mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMerge\loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
T0
*
N
в
6loss_2/output_1_loss/broadcast_weights/ones_like/ShapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsN^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
╦
6loss_2/output_1_loss/broadcast_weights/ones_like/ConstConstN^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
valueB
 *  А?*
dtype0
ш
0loss_2/output_1_loss/broadcast_weights/ones_likeFill6loss_2/output_1_loss/broadcast_weights/ones_like/Shape6loss_2/output_1_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0*

index_type0
и
&loss_2/output_1_loss/broadcast_weightsMuloutput_1_sample_weights_20loss_2/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
╙
loss_2/output_1_loss/MulMul\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&loss_2/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0
d
loss_2/output_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
У
loss_2/output_1_loss/SumSumloss_2/output_1_loss/Mulloss_2/output_1_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
f
loss_2/output_1_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
е
loss_2/output_1_loss/Sum_1Sum&loss_2/output_1_loss/broadcast_weightsloss_2/output_1_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
В
loss_2/output_1_loss/div_no_nanDivNoNanloss_2/output_1_loss/Sumloss_2/output_1_loss/Sum_1*
T0*
_output_shapes
: 
_
loss_2/output_1_loss/Const_2Const*
_output_shapes
: *
valueB *
dtype0
Ю
loss_2/output_1_loss/MeanMeanloss_2/output_1_loss/div_no_nanloss_2/output_1_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Q
loss_2/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
[

loss_2/mulMulloss_2/mul/xloss_2/output_1_loss/Mean*
T0*
_output_shapes
: 
З
metrics_2/acc/CastCastoutput_1_target_2*0
_output_shapes
:                  *
Truncate( *

SrcT0*

DstT0
В
metrics_2/acc/SqueezeSqueezemetrics_2/acc/Cast*#
_output_shapes
:         *
T0*
squeeze_dims

         
i
metrics_2/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ц
metrics_2/acc/ArgMaxArgMax	Softmax_2metrics_2/acc/ArgMax/dimension*#
_output_shapes
:         *

Tidx0*
T0*
output_type0	

metrics_2/acc/Cast_1Castmetrics_2/acc/ArgMax*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
w
metrics_2/acc/EqualEqualmetrics_2/acc/Squeezemetrics_2/acc/Cast_1*#
_output_shapes
:         *
T0
~
metrics_2/acc/Cast_2Castmetrics_2/acc/Equal*
Truncate( *

SrcT0
*

DstT0*#
_output_shapes
:         
a
metrics_2/acc/SizeSizemetrics_2/acc/Cast_2*
_output_shapes
: *
T0*
out_type0
p
metrics_2/acc/Cast_3Castmetrics_2/acc/Size*
Truncate( *

SrcT0*

DstT0*
_output_shapes
: 
]
metrics_2/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Б
metrics_2/acc/SumSummetrics_2/acc/Cast_2metrics_2/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
!metrics_2/acc/AssignAddVariableOpAssignAddVariableOptotal_2metrics_2/acc/Sum*
dtype0
А
metrics_2/acc/ReadVariableOpReadVariableOptotal_2"^metrics_2/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
Е
#metrics_2/acc/AssignAddVariableOp_1AssignAddVariableOpcount_2metrics_2/acc/Cast_3^metrics_2/acc/ReadVariableOp*
dtype0
г
metrics_2/acc/ReadVariableOp_1ReadVariableOpcount_2$^metrics_2/acc/AssignAddVariableOp_1^metrics_2/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
И
'metrics_2/acc/div_no_nan/ReadVariableOpReadVariableOptotal_2^metrics_2/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
К
)metrics_2/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_2^metrics_2/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Щ
metrics_2/acc/div_no_nanDivNoNan'metrics_2/acc/div_no_nan/ReadVariableOp)metrics_2/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Г
metrics_2/acc/Squeeze_1Squeezeoutput_1_target_2*
squeeze_dims

         *#
_output_shapes
:         *
T0
k
 metrics_2/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
metrics_2/acc/ArgMax_1ArgMax	Softmax_2 metrics_2/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
Б
metrics_2/acc/Cast_4Castmetrics_2/acc/ArgMax_1*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
{
metrics_2/acc/Equal_1Equalmetrics_2/acc/Squeeze_1metrics_2/acc/Cast_4*#
_output_shapes
:         *
T0
А
metrics_2/acc/Cast_5Castmetrics_2/acc/Equal_1*
Truncate( *

SrcT0
*

DstT0*#
_output_shapes
:         
_
metrics_2/acc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Е
metrics_2/acc/MeanMeanmetrics_2/acc/Cast_5metrics_2/acc/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Б
training_4/Adam/gradients/ShapeConst*
valueB *
_class
loc:@loss_2/mul*
dtype0*
_output_shapes
: 
З
#training_4/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
_class
loc:@loss_2/mul*
dtype0
╛
training_4/Adam/gradients/FillFilltraining_4/Adam/gradients/Shape#training_4/Adam/gradients/grad_ys_0*
_class
loc:@loss_2/mul*
_output_shapes
: *
T0*

index_type0
п
-training_4/Adam/gradients/loss_2/mul_grad/MulMultraining_4/Adam/gradients/Fillloss_2/output_1_loss/Mean*
_class
loc:@loss_2/mul*
_output_shapes
: *
T0
д
/training_4/Adam/gradients/loss_2/mul_grad/Mul_1Multraining_4/Adam/gradients/Fillloss_2/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss_2/mul
╖
Ftraining_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB *,
_class"
 loc:@loss_2/output_1_loss/Mean*
dtype0*
_output_shapes
: 
б
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/ReshapeReshape/training_4/Adam/gradients/loss_2/mul_grad/Mul_1Ftraining_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape/shape*
Tshape0*,
_class"
 loc:@loss_2/output_1_loss/Mean*
_output_shapes
: *
T0
п
>training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/ConstConst*
valueB *,
_class"
 loc:@loss_2/output_1_loss/Mean*
dtype0*
_output_shapes
: 
и
=training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/TileTile@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape>training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const*

Tmultiples0*,
_class"
 loc:@loss_2/output_1_loss/Mean*
_output_shapes
: *
T0
│
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const_1Const*
valueB
 *  А?*,
_class"
 loc:@loss_2/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ы
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivRealDiv=training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Tile@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const_1*
T0*,
_class"
 loc:@loss_2/output_1_loss/Mean*
_output_shapes
: 
╗
Dtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ShapeConst*
valueB *2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
╜
Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1Const*
valueB *2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
ь
Ttraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ShapeFtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1*2
_output_shapes 
:         :         *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
И
Itraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivloss_2/output_1_loss/Sum_1*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *
T0
▄
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/SumSumItraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nanTtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
╛
Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ReshapeReshapeBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/SumDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape*
_output_shapes
: *
T0*
Tshape0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
╕
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/NegNegloss_2/output_1_loss/Sum*
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: 
М
Ktraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNanBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Negloss_2/output_1_loss/Sum_1*
_output_shapes
: *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
Х
Ktraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanKtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_1loss_2/output_1_loss/Sum_1*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *
T0
н
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/mulMul@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivKtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_2*
_output_shapes
: *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
┘
Dtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Sum_1SumBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/mulVtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
─
Htraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Reshape_1ReshapeDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Sum_1Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
╝
Etraining_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss_2/output_1_loss/Sum*
dtype0*
_output_shapes
:
╣
?training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/ReshapeReshapeFtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ReshapeEtraining_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*+
_class!
loc:@loss_2/output_1_loss/Sum
┬
=training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/ShapeShapeloss_2/output_1_loss/Mul*+
_class!
loc:@loss_2/output_1_loss/Sum*
_output_shapes
:*
T0*
out_type0
▒
<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/TileTile?training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape=training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss_2/output_1_loss/Sum*#
_output_shapes
:         
Ж
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ShapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*+
_class!
loc:@loss_2/output_1_loss/Mul*
_output_shapes
:*
T0*
out_type0
╥
?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1Shape&loss_2/output_1_loss/broadcast_weights*
_output_shapes
:*
T0*
out_type0*+
_class!
loc:@loss_2/output_1_loss/Mul
╨
Mtraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1*+
_class!
loc:@loss_2/output_1_loss/Mul*2
_output_shapes 
:         :         *
T0
Г
;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/MulMul<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Tile&loss_2/output_1_loss/broadcast_weights*+
_class!
loc:@loss_2/output_1_loss/Mul*#
_output_shapes
:         *
T0
╗
;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/SumSum;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/MulMtraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgs*+
_class!
loc:@loss_2/output_1_loss/Mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
п
?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ReshapeReshape;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0*+
_class!
loc:@loss_2/output_1_loss/Mul
╗
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Mul_1Mul\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Tile*
T0*+
_class!
loc:@loss_2/output_1_loss/Mul*#
_output_shapes
:         
┴
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum_1Sum=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Mul_1Otraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgs:1*+
_class!
loc:@loss_2/output_1_loss/Mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╡
Atraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Reshape_1Reshape=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum_1?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1*
T0*
Tshape0*+
_class!
loc:@loss_2/output_1_loss/Mul*#
_output_shapes
:         
┤
$training_4/Adam/gradients/zeros_like	ZerosLike^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
┘
Лtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
*
T0
╟
Кtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0*
_output_shapes
: 
Т
Жtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ReshapeКtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         *
T0
└
training_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulЖtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsЛtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

┐
Ctraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_5*1
_class'
%#loc:@loss_2/output_1_loss/Reshape_1*
_output_shapes
:*
T0*
out_type0
Й
Etraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ReshapeReshapetraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulCtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Shape*'
_output_shapes
:         
*
T0*
Tshape0*1
_class'
%#loc:@loss_2/output_1_loss/Reshape_1
ф
4training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Reshape*
T0*
data_formatNHWC*
_class
loc:@BiasAdd_5*
_output_shapes
:

Н
.training_4/Adam/gradients/MatMul_5_grad/MatMulMatMulEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ReshapeMatMul_5/ReadVariableOp*
_class
loc:@MatMul_5*'
_output_shapes
:         *
transpose_b(*
T0*
transpose_a( 
√
0training_4/Adam/gradients/MatMul_5_grad/MatMul_1MatMulcond_2/MergeEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@MatMul_5*
_output_shapes

:
*
transpose_b( *
T0*
transpose_a(
с
5training_4/Adam/gradients/cond_2/Merge_grad/cond_gradSwitch.training_4/Adam/gradients/MatMul_5_grad/MatMulcond_2/pred_id*
T0*
_class
loc:@MatMul_5*:
_output_shapes(
&:         :         
┤
7training_4/Adam/gradients/cond_2/dropout/mul_grad/ShapeShapecond_2/dropout/truediv*
T0*
out_type0*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:
┤
9training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1Shapecond_2/dropout/Floor*
T0*
out_type0*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:
╕
Gtraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape9training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0*%
_class
loc:@cond_2/dropout/mul
ф
5training_4/Adam/gradients/cond_2/dropout/mul_grad/MulMul7training_4/Adam/gradients/cond_2/Merge_grad/cond_grad:1cond_2/dropout/Floor*'
_output_shapes
:         *
T0*%
_class
loc:@cond_2/dropout/mul
г
5training_4/Adam/gradients/cond_2/dropout/mul_grad/SumSum5training_4/Adam/gradients/cond_2/dropout/mul_grad/MulGtraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ы
9training_4/Adam/gradients/cond_2/dropout/mul_grad/ReshapeReshape5training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum7training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0*%
_class
loc:@cond_2/dropout/mul
ш
7training_4/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Mulcond_2/dropout/truediv7training_4/Adam/gradients/cond_2/Merge_grad/cond_grad:1*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:         *
T0
й
7training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum_1Sum7training_4/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Itraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*%
_class
loc:@cond_2/dropout/mul
б
;training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshape_1Reshape7training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum_19training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*
Tshape0*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:         
в
 training_4/Adam/gradients/SwitchSwitchRelu_2cond_2/pred_id*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         *
T0
Я
"training_4/Adam/gradients/IdentityIdentity"training_4/Adam/gradients/Switch:1*
T0*
_class
loc:@Relu_2*'
_output_shapes
:         
Ю
!training_4/Adam/gradients/Shape_1Shape"training_4/Adam/gradients/Switch:1*
T0*
out_type0*
_class
loc:@Relu_2*
_output_shapes
:
к
%training_4/Adam/gradients/zeros/ConstConst#^training_4/Adam/gradients/Identity*
valueB
 *    *
_class
loc:@Relu_2*
dtype0*
_output_shapes
: 
╨
training_4/Adam/gradients/zerosFill!training_4/Adam/gradients/Shape_1%training_4/Adam/gradients/zeros/Const*
T0*

index_type0*
_class
loc:@Relu_2*'
_output_shapes
:         
°
?training_4/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradMerge5training_4/Adam/gradients/cond_2/Merge_grad/cond_gradtraining_4/Adam/gradients/zeros*
_class
loc:@Relu_2*)
_output_shapes
:         : *
N*
T0
├
;training_4/Adam/gradients/cond_2/dropout/truediv_grad/ShapeShapecond_2/dropout/Shape/Switch:1*)
_class
loc:@cond_2/dropout/truediv*
_output_shapes
:*
T0*
out_type0
л
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1Const*
valueB *)
_class
loc:@cond_2/dropout/truediv*
dtype0*
_output_shapes
: 
╚
Ktraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1*2
_output_shapes 
:         :         *
T0*)
_class
loc:@cond_2/dropout/truediv
Ї
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDivRealDiv9training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshapecond_2/dropout/sub*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         *
T0
╖
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/SumSum=training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDivKtraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgs*)
_class
loc:@cond_2/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
л
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/ReshapeReshape9training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0*)
_class
loc:@cond_2/dropout/truediv
╝
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/NegNegcond_2/dropout/Shape/Switch:1*
T0*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         
Ў
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_1RealDiv9training_4/Adam/gradients/cond_2/dropout/truediv_grad/Negcond_2/dropout/sub*
T0*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         
№
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_2RealDiv?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_1cond_2/dropout/sub*
T0*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         
Щ
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/mulMul9training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshape?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_2*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         *
T0
╖
;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum_1Sum9training_4/Adam/gradients/cond_2/dropout/truediv_grad/mulMtraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgs:1*)
_class
loc:@cond_2/dropout/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
а
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/Reshape_1Reshape;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum_1=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*)
_class
loc:@cond_2/dropout/truediv
д
"training_4/Adam/gradients/Switch_1SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         
б
$training_4/Adam/gradients/Identity_1Identity"training_4/Adam/gradients/Switch_1*
_class
loc:@Relu_2*'
_output_shapes
:         *
T0
Ю
!training_4/Adam/gradients/Shape_2Shape"training_4/Adam/gradients/Switch_1*
T0*
out_type0*
_class
loc:@Relu_2*
_output_shapes
:
о
'training_4/Adam/gradients/zeros_1/ConstConst%^training_4/Adam/gradients/Identity_1*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Relu_2
╘
!training_4/Adam/gradients/zeros_1Fill!training_4/Adam/gradients/Shape_2'training_4/Adam/gradients/zeros_1/Const*
_class
loc:@Relu_2*'
_output_shapes
:         *
T0*

index_type0
З
Dtraining_4/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_gradMerge!training_4/Adam/gradients/zeros_1=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Reshape*)
_output_shapes
:         : *
N*
T0*
_class
loc:@Relu_2
Г
training_4/Adam/gradients/AddNAddN?training_4/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradDtraining_4/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_grad*
N*
T0*
_class
loc:@Relu_2*'
_output_shapes
:         
п
.training_4/Adam/gradients/Relu_2_grad/ReluGradReluGradtraining_4/Adam/gradients/AddNRelu_2*
T0*
_class
loc:@Relu_2*'
_output_shapes
:         
═
4training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad.training_4/Adam/gradients/Relu_2_grad/ReluGrad*
_class
loc:@BiasAdd_4*
_output_shapes
:*
T0*
data_formatNHWC
ў
.training_4/Adam/gradients/MatMul_4_grad/MatMulMatMul.training_4/Adam/gradients/Relu_2_grad/ReluGradMatMul_4/ReadVariableOp*
_class
loc:@MatMul_4*(
_output_shapes
:         Р*
transpose_b(*
T0*
transpose_a( 
т
0training_4/Adam/gradients/MatMul_4_grad/MatMul_1MatMul	Reshape_2.training_4/Adam/gradients/Relu_2_grad/ReluGrad*
_output_shapes
:	Р*
transpose_b( *
T0*
transpose_a(*
_class
loc:@MatMul_4
W
training_4/Adam/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R
q
#training_4/Adam/AssignAddVariableOpAssignAddVariableOpAdam_1/iterationstraining_4/Adam/Const*
dtype0	
О
training_4/Adam/ReadVariableOpReadVariableOpAdam_1/iterations$^training_4/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
О
#training_4/Adam/Cast/ReadVariableOpReadVariableOpAdam_1/iterations^training_4/Adam/ReadVariableOp*
dtype0	*
_output_shapes
: 
Б
training_4/Adam/CastCast#training_4/Adam/Cast/ReadVariableOp*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	
h
"training_4/Adam/Pow/ReadVariableOpReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
u
training_4/Adam/PowPow"training_4/Adam/Pow/ReadVariableOptraining_4/Adam/Cast*
_output_shapes
: *
T0
Z
training_4/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
training_4/Adam/subSubtraining_4/Adam/sub/xtraining_4/Adam/Pow*
_output_shapes
: *
T0
\
training_4/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_2Const*
_output_shapes
: *
valueB
 *  А*
dtype0

%training_4/Adam/clip_by_value/MinimumMinimumtraining_4/Adam/subtraining_4/Adam/Const_2*
_output_shapes
: *
T0
Й
training_4/Adam/clip_by_valueMaximum%training_4/Adam/clip_by_value/Minimumtraining_4/Adam/Const_1*
_output_shapes
: *
T0
\
training_4/Adam/SqrtSqrttraining_4/Adam/clip_by_value*
T0*
_output_shapes
: 
j
$training_4/Adam/Pow_1/ReadVariableOpReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
y
training_4/Adam/Pow_1Pow$training_4/Adam/Pow_1/ReadVariableOptraining_4/Adam/Cast*
_output_shapes
: *
T0
\
training_4/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
m
training_4/Adam/sub_1Subtraining_4/Adam/sub_1/xtraining_4/Adam/Pow_1*
T0*
_output_shapes
: 
p
training_4/Adam/truedivRealDivtraining_4/Adam/Sqrttraining_4/Adam/sub_1*
_output_shapes
: *
T0
b
 training_4/Adam/ReadVariableOp_1ReadVariableOp	Adam_1/lr*
_output_shapes
: *
dtype0
v
training_4/Adam/mulMul training_4/Adam/ReadVariableOp_1training_4/Adam/truediv*
_output_shapes
: *
T0
v
%training_4/Adam/zeros/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
`
training_4/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
training_4/Adam/zerosFill%training_4/Adam/zeros/shape_as_tensortraining_4/Adam/zeros/Const*

index_type0*
_output_shapes
:	Р*
T0
╦
training_4/Adam/VariableVarHandleOp*
dtype0*
shape:	Р*
	container *+
_class!
loc:@training_4/Adam/Variable*
_output_shapes
: *)
shared_nametraining_4/Adam/Variable
Б
9training_4/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable*
_output_shapes
: 
Ю
training_4/Adam/Variable/AssignAssignVariableOptraining_4/Adam/Variabletraining_4/Adam/zeros*+
_class!
loc:@training_4/Adam/Variable*
dtype0
│
,training_4/Adam/Variable/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable*+
_class!
loc:@training_4/Adam/Variable*
dtype0*
_output_shapes
:	Р
d
training_4/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
╠
training_4/Adam/Variable_1VarHandleOp*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_1*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_1*
dtype0
Е
;training_4/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_1*
_output_shapes
: 
ж
!training_4/Adam/Variable_1/AssignAssignVariableOptraining_4/Adam/Variable_1training_4/Adam/zeros_1*-
_class#
!loc:@training_4/Adam/Variable_1*
dtype0
┤
.training_4/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_1*-
_class#
!loc:@training_4/Adam/Variable_1*
dtype0*
_output_shapes
:
l
training_4/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

╨
training_4/Adam/Variable_2VarHandleOp*
shape
:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_2*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_2*
dtype0
Е
;training_4/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_2*
_output_shapes
: 
ж
!training_4/Adam/Variable_2/AssignAssignVariableOptraining_4/Adam/Variable_2training_4/Adam/zeros_2*-
_class#
!loc:@training_4/Adam/Variable_2*
dtype0
╕
.training_4/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_2*-
_class#
!loc:@training_4/Adam/Variable_2*
dtype0*
_output_shapes

:

d
training_4/Adam/zeros_3Const*
dtype0*
_output_shapes
:
*
valueB
*    
╠
training_4/Adam/Variable_3VarHandleOp*
shape:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_3*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_3*
dtype0
Е
;training_4/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_3*
_output_shapes
: 
ж
!training_4/Adam/Variable_3/AssignAssignVariableOptraining_4/Adam/Variable_3training_4/Adam/zeros_3*-
_class#
!loc:@training_4/Adam/Variable_3*
dtype0
┤
.training_4/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_3*-
_class#
!loc:@training_4/Adam/Variable_3*
dtype0*
_output_shapes
:

x
'training_4/Adam/zeros_4/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
b
training_4/Adam/zeros_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
г
training_4/Adam/zeros_4Fill'training_4/Adam/zeros_4/shape_as_tensortraining_4/Adam/zeros_4/Const*
_output_shapes
:	Р*
T0*

index_type0
╤
training_4/Adam/Variable_4VarHandleOp*
dtype0*
shape:	Р*
	container *-
_class#
!loc:@training_4/Adam/Variable_4*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_4
Е
;training_4/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_4*
_output_shapes
: 
ж
!training_4/Adam/Variable_4/AssignAssignVariableOptraining_4/Adam/Variable_4training_4/Adam/zeros_4*-
_class#
!loc:@training_4/Adam/Variable_4*
dtype0
╣
.training_4/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_4*-
_class#
!loc:@training_4/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
d
training_4/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
╠
training_4/Adam/Variable_5VarHandleOp*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_5*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_5*
dtype0
Е
;training_4/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_5*
_output_shapes
: 
ж
!training_4/Adam/Variable_5/AssignAssignVariableOptraining_4/Adam/Variable_5training_4/Adam/zeros_5*-
_class#
!loc:@training_4/Adam/Variable_5*
dtype0
┤
.training_4/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_5*-
_class#
!loc:@training_4/Adam/Variable_5*
dtype0*
_output_shapes
:
l
training_4/Adam/zeros_6Const*
dtype0*
_output_shapes

:
*
valueB
*    
╨
training_4/Adam/Variable_6VarHandleOp*
dtype0*
shape
:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_6*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_6
Е
;training_4/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_6*
_output_shapes
: 
ж
!training_4/Adam/Variable_6/AssignAssignVariableOptraining_4/Adam/Variable_6training_4/Adam/zeros_6*-
_class#
!loc:@training_4/Adam/Variable_6*
dtype0
╕
.training_4/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_6*-
_class#
!loc:@training_4/Adam/Variable_6*
dtype0*
_output_shapes

:

d
training_4/Adam/zeros_7Const*
valueB
*    *
dtype0*
_output_shapes
:

╠
training_4/Adam/Variable_7VarHandleOp*-
_class#
!loc:@training_4/Adam/Variable_7*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_7*
dtype0*
shape:
*
	container 
Е
;training_4/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_7*
_output_shapes
: 
ж
!training_4/Adam/Variable_7/AssignAssignVariableOptraining_4/Adam/Variable_7training_4/Adam/zeros_7*-
_class#
!loc:@training_4/Adam/Variable_7*
dtype0
┤
.training_4/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_7*-
_class#
!loc:@training_4/Adam/Variable_7*
dtype0*
_output_shapes
:

q
'training_4/Adam/zeros_8/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training_4/Adam/zeros_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ю
training_4/Adam/zeros_8Fill'training_4/Adam/zeros_8/shape_as_tensortraining_4/Adam/zeros_8/Const*
_output_shapes
:*
T0*

index_type0
╠
training_4/Adam/Variable_8VarHandleOp*
dtype0*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_8*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_8
Е
;training_4/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_8*
_output_shapes
: 
ж
!training_4/Adam/Variable_8/AssignAssignVariableOptraining_4/Adam/Variable_8training_4/Adam/zeros_8*-
_class#
!loc:@training_4/Adam/Variable_8*
dtype0
┤
.training_4/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_8*-
_class#
!loc:@training_4/Adam/Variable_8*
dtype0*
_output_shapes
:
q
'training_4/Adam/zeros_9/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
b
training_4/Adam/zeros_9/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ю
training_4/Adam/zeros_9Fill'training_4/Adam/zeros_9/shape_as_tensortraining_4/Adam/zeros_9/Const*

index_type0*
_output_shapes
:*
T0
╠
training_4/Adam/Variable_9VarHandleOp*-
_class#
!loc:@training_4/Adam/Variable_9*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_9*
dtype0*
shape:*
	container 
Е
;training_4/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_9*
_output_shapes
: 
ж
!training_4/Adam/Variable_9/AssignAssignVariableOptraining_4/Adam/Variable_9training_4/Adam/zeros_9*-
_class#
!loc:@training_4/Adam/Variable_9*
dtype0
┤
.training_4/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_9*-
_class#
!loc:@training_4/Adam/Variable_9*
dtype0*
_output_shapes
:
r
(training_4/Adam/zeros_10/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
c
training_4/Adam/zeros_10/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
б
training_4/Adam/zeros_10Fill(training_4/Adam/zeros_10/shape_as_tensortraining_4/Adam/zeros_10/Const*
_output_shapes
:*
T0*

index_type0
╧
training_4/Adam/Variable_10VarHandleOp*
dtype0*
shape:*
	container *.
_class$
" loc:@training_4/Adam/Variable_10*
_output_shapes
: *,
shared_nametraining_4/Adam/Variable_10
З
<training_4/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_10*
_output_shapes
: 
к
"training_4/Adam/Variable_10/AssignAssignVariableOptraining_4/Adam/Variable_10training_4/Adam/zeros_10*.
_class$
" loc:@training_4/Adam/Variable_10*
dtype0
╖
/training_4/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_10*.
_class$
" loc:@training_4/Adam/Variable_10*
dtype0*
_output_shapes
:
r
(training_4/Adam/zeros_11/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
c
training_4/Adam/zeros_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
б
training_4/Adam/zeros_11Fill(training_4/Adam/zeros_11/shape_as_tensortraining_4/Adam/zeros_11/Const*
_output_shapes
:*
T0*

index_type0
╧
training_4/Adam/Variable_11VarHandleOp*
shape:*
	container *.
_class$
" loc:@training_4/Adam/Variable_11*
_output_shapes
: *,
shared_nametraining_4/Adam/Variable_11*
dtype0
З
<training_4/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_11*
_output_shapes
: 
к
"training_4/Adam/Variable_11/AssignAssignVariableOptraining_4/Adam/Variable_11training_4/Adam/zeros_11*
dtype0*.
_class$
" loc:@training_4/Adam/Variable_11
╖
/training_4/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_11*.
_class$
" loc:@training_4/Adam/Variable_11*
dtype0*
_output_shapes
:
f
 training_4/Adam/ReadVariableOp_2ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
~
$training_4/Adam/mul_1/ReadVariableOpReadVariableOptraining_4/Adam/Variable*
dtype0*
_output_shapes
:	Р
О
training_4/Adam/mul_1Mul training_4/Adam/ReadVariableOp_2$training_4/Adam/mul_1/ReadVariableOp*
T0*
_output_shapes
:	Р
f
 training_4/Adam/ReadVariableOp_3ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
x
training_4/Adam/sub_2Subtraining_4/Adam/sub_2/x training_4/Adam/ReadVariableOp_3*
_output_shapes
: *
T0
П
training_4/Adam/mul_2Multraining_4/Adam/sub_20training_4/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes
:	Р*
T0
r
training_4/Adam/addAddtraining_4/Adam/mul_1training_4/Adam/mul_2*
T0*
_output_shapes
:	Р
f
 training_4/Adam/ReadVariableOp_4ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
А
$training_4/Adam/mul_3/ReadVariableOpReadVariableOptraining_4/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
О
training_4/Adam/mul_3Mul training_4/Adam/ReadVariableOp_4$training_4/Adam/mul_3/ReadVariableOp*
_output_shapes
:	Р*
T0
f
 training_4/Adam/ReadVariableOp_5ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
x
training_4/Adam/sub_3Subtraining_4/Adam/sub_3/x training_4/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 
|
training_4/Adam/SquareSquare0training_4/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes
:	Р*
T0
u
training_4/Adam/mul_4Multraining_4/Adam/sub_3training_4/Adam/Square*
_output_shapes
:	Р*
T0
t
training_4/Adam/add_1Addtraining_4/Adam/mul_3training_4/Adam/mul_4*
T0*
_output_shapes
:	Р
p
training_4/Adam/mul_5Multraining_4/Adam/multraining_4/Adam/add*
_output_shapes
:	Р*
T0
\
training_4/Adam/Const_3Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_4Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
М
'training_4/Adam/clip_by_value_1/MinimumMinimumtraining_4/Adam/add_1training_4/Adam/Const_4*
T0*
_output_shapes
:	Р
Ц
training_4/Adam/clip_by_value_1Maximum'training_4/Adam/clip_by_value_1/Minimumtraining_4/Adam/Const_3*
_output_shapes
:	Р*
T0
i
training_4/Adam/Sqrt_1Sqrttraining_4/Adam/clip_by_value_1*
_output_shapes
:	Р*
T0
\
training_4/Adam/add_2/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
w
training_4/Adam/add_2Addtraining_4/Adam/Sqrt_1training_4/Adam/add_2/y*
_output_shapes
:	Р*
T0
|
training_4/Adam/truediv_1RealDivtraining_4/Adam/mul_5training_4/Adam/add_2*
_output_shapes
:	Р*
T0
p
 training_4/Adam/ReadVariableOp_6ReadVariableOpdense_4/kernel*
dtype0*
_output_shapes
:	Р
Г
training_4/Adam/sub_4Sub training_4/Adam/ReadVariableOp_6training_4/Adam/truediv_1*
_output_shapes
:	Р*
T0
p
 training_4/Adam/AssignVariableOpAssignVariableOptraining_4/Adam/Variabletraining_4/Adam/add*
dtype0
Э
 training_4/Adam/ReadVariableOp_7ReadVariableOptraining_4/Adam/Variable!^training_4/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Р
v
"training_4/Adam/AssignVariableOp_1AssignVariableOptraining_4/Adam/Variable_4training_4/Adam/add_1*
dtype0
б
 training_4/Adam/ReadVariableOp_8ReadVariableOptraining_4/Adam/Variable_4#^training_4/Adam/AssignVariableOp_1*
dtype0*
_output_shapes
:	Р
j
"training_4/Adam/AssignVariableOp_2AssignVariableOpdense_4/kerneltraining_4/Adam/sub_4*
dtype0
Х
 training_4/Adam/ReadVariableOp_9ReadVariableOpdense_4/kernel#^training_4/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Р
g
!training_4/Adam/ReadVariableOp_10ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
{
$training_4/Adam/mul_6/ReadVariableOpReadVariableOptraining_4/Adam/Variable_1*
dtype0*
_output_shapes
:
К
training_4/Adam/mul_6Mul!training_4/Adam/ReadVariableOp_10$training_4/Adam/mul_6/ReadVariableOp*
_output_shapes
:*
T0
g
!training_4/Adam/ReadVariableOp_11ReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
\
training_4/Adam/sub_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
y
training_4/Adam/sub_5Subtraining_4/Adam/sub_5/x!training_4/Adam/ReadVariableOp_11*
_output_shapes
: *
T0
О
training_4/Adam/mul_7Multraining_4/Adam/sub_54training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training_4/Adam/add_3Addtraining_4/Adam/mul_6training_4/Adam/mul_7*
T0*
_output_shapes
:
g
!training_4/Adam/ReadVariableOp_12ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
{
$training_4/Adam/mul_8/ReadVariableOpReadVariableOptraining_4/Adam/Variable_5*
dtype0*
_output_shapes
:
К
training_4/Adam/mul_8Mul!training_4/Adam/ReadVariableOp_12$training_4/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes
:
g
!training_4/Adam/ReadVariableOp_13ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_6Subtraining_4/Adam/sub_6/x!training_4/Adam/ReadVariableOp_13*
_output_shapes
: *
T0
}
training_4/Adam/Square_1Square4training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:*
T0
r
training_4/Adam/mul_9Multraining_4/Adam/sub_6training_4/Adam/Square_1*
_output_shapes
:*
T0
o
training_4/Adam/add_4Addtraining_4/Adam/mul_8training_4/Adam/mul_9*
T0*
_output_shapes
:
n
training_4/Adam/mul_10Multraining_4/Adam/multraining_4/Adam/add_3*
_output_shapes
:*
T0
\
training_4/Adam/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *    
\
training_4/Adam/Const_6Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
З
'training_4/Adam/clip_by_value_2/MinimumMinimumtraining_4/Adam/add_4training_4/Adam/Const_6*
_output_shapes
:*
T0
С
training_4/Adam/clip_by_value_2Maximum'training_4/Adam/clip_by_value_2/Minimumtraining_4/Adam/Const_5*
_output_shapes
:*
T0
d
training_4/Adam/Sqrt_2Sqrttraining_4/Adam/clip_by_value_2*
_output_shapes
:*
T0
\
training_4/Adam/add_5/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
r
training_4/Adam/add_5Addtraining_4/Adam/Sqrt_2training_4/Adam/add_5/y*
_output_shapes
:*
T0
x
training_4/Adam/truediv_2RealDivtraining_4/Adam/mul_10training_4/Adam/add_5*
_output_shapes
:*
T0
j
!training_4/Adam/ReadVariableOp_14ReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:

training_4/Adam/sub_7Sub!training_4/Adam/ReadVariableOp_14training_4/Adam/truediv_2*
_output_shapes
:*
T0
v
"training_4/Adam/AssignVariableOp_3AssignVariableOptraining_4/Adam/Variable_1training_4/Adam/add_3*
dtype0
Э
!training_4/Adam/ReadVariableOp_15ReadVariableOptraining_4/Adam/Variable_1#^training_4/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:
v
"training_4/Adam/AssignVariableOp_4AssignVariableOptraining_4/Adam/Variable_5training_4/Adam/add_4*
dtype0
Э
!training_4/Adam/ReadVariableOp_16ReadVariableOptraining_4/Adam/Variable_5#^training_4/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
:
h
"training_4/Adam/AssignVariableOp_5AssignVariableOpdense_4/biastraining_4/Adam/sub_7*
dtype0
П
!training_4/Adam/ReadVariableOp_17ReadVariableOpdense_4/bias#^training_4/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:
g
!training_4/Adam/ReadVariableOp_18ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
А
%training_4/Adam/mul_11/ReadVariableOpReadVariableOptraining_4/Adam/Variable_2*
_output_shapes

:
*
dtype0
Р
training_4/Adam/mul_11Mul!training_4/Adam/ReadVariableOp_18%training_4/Adam/mul_11/ReadVariableOp*
_output_shapes

:
*
T0
g
!training_4/Adam/ReadVariableOp_19ReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
\
training_4/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_8Subtraining_4/Adam/sub_8/x!training_4/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
П
training_4/Adam/mul_12Multraining_4/Adam/sub_80training_4/Adam/gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:
*
T0
u
training_4/Adam/add_6Addtraining_4/Adam/mul_11training_4/Adam/mul_12*
_output_shapes

:
*
T0
g
!training_4/Adam/ReadVariableOp_20ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
А
%training_4/Adam/mul_13/ReadVariableOpReadVariableOptraining_4/Adam/Variable_6*
dtype0*
_output_shapes

:

Р
training_4/Adam/mul_13Mul!training_4/Adam/ReadVariableOp_20%training_4/Adam/mul_13/ReadVariableOp*
_output_shapes

:
*
T0
g
!training_4/Adam/ReadVariableOp_21ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_9Subtraining_4/Adam/sub_9/x!training_4/Adam/ReadVariableOp_21*
_output_shapes
: *
T0
}
training_4/Adam/Square_2Square0training_4/Adam/gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:
*
T0
w
training_4/Adam/mul_14Multraining_4/Adam/sub_9training_4/Adam/Square_2*
T0*
_output_shapes

:

u
training_4/Adam/add_7Addtraining_4/Adam/mul_13training_4/Adam/mul_14*
T0*
_output_shapes

:

r
training_4/Adam/mul_15Multraining_4/Adam/multraining_4/Adam/add_6*
_output_shapes

:
*
T0
\
training_4/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_8Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Л
'training_4/Adam/clip_by_value_3/MinimumMinimumtraining_4/Adam/add_7training_4/Adam/Const_8*
_output_shapes

:
*
T0
Х
training_4/Adam/clip_by_value_3Maximum'training_4/Adam/clip_by_value_3/Minimumtraining_4/Adam/Const_7*
_output_shapes

:
*
T0
h
training_4/Adam/Sqrt_3Sqrttraining_4/Adam/clip_by_value_3*
_output_shapes

:
*
T0
\
training_4/Adam/add_8/yConst*
_output_shapes
: *
valueB
 *Х┐╓3*
dtype0
v
training_4/Adam/add_8Addtraining_4/Adam/Sqrt_3training_4/Adam/add_8/y*
_output_shapes

:
*
T0
|
training_4/Adam/truediv_3RealDivtraining_4/Adam/mul_15training_4/Adam/add_8*
T0*
_output_shapes

:

p
!training_4/Adam/ReadVariableOp_22ReadVariableOpdense_5/kernel*
dtype0*
_output_shapes

:

Д
training_4/Adam/sub_10Sub!training_4/Adam/ReadVariableOp_22training_4/Adam/truediv_3*
_output_shapes

:
*
T0
v
"training_4/Adam/AssignVariableOp_6AssignVariableOptraining_4/Adam/Variable_2training_4/Adam/add_6*
dtype0
б
!training_4/Adam/ReadVariableOp_23ReadVariableOptraining_4/Adam/Variable_2#^training_4/Adam/AssignVariableOp_6*
dtype0*
_output_shapes

:

v
"training_4/Adam/AssignVariableOp_7AssignVariableOptraining_4/Adam/Variable_6training_4/Adam/add_7*
dtype0
б
!training_4/Adam/ReadVariableOp_24ReadVariableOptraining_4/Adam/Variable_6#^training_4/Adam/AssignVariableOp_7*
dtype0*
_output_shapes

:

k
"training_4/Adam/AssignVariableOp_8AssignVariableOpdense_5/kerneltraining_4/Adam/sub_10*
dtype0
Х
!training_4/Adam/ReadVariableOp_25ReadVariableOpdense_5/kernel#^training_4/Adam/AssignVariableOp_8*
_output_shapes

:
*
dtype0
g
!training_4/Adam/ReadVariableOp_26ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
|
%training_4/Adam/mul_16/ReadVariableOpReadVariableOptraining_4/Adam/Variable_3*
dtype0*
_output_shapes
:

М
training_4/Adam/mul_16Mul!training_4/Adam/ReadVariableOp_26%training_4/Adam/mul_16/ReadVariableOp*
T0*
_output_shapes
:

g
!training_4/Adam/ReadVariableOp_27ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
]
training_4/Adam/sub_11/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
{
training_4/Adam/sub_11Subtraining_4/Adam/sub_11/x!training_4/Adam/ReadVariableOp_27*
_output_shapes
: *
T0
Р
training_4/Adam/mul_17Multraining_4/Adam/sub_114training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:

q
training_4/Adam/add_9Addtraining_4/Adam/mul_16training_4/Adam/mul_17*
_output_shapes
:
*
T0
g
!training_4/Adam/ReadVariableOp_28ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
|
%training_4/Adam/mul_18/ReadVariableOpReadVariableOptraining_4/Adam/Variable_7*
dtype0*
_output_shapes
:

М
training_4/Adam/mul_18Mul!training_4/Adam/ReadVariableOp_28%training_4/Adam/mul_18/ReadVariableOp*
_output_shapes
:
*
T0
g
!training_4/Adam/ReadVariableOp_29ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
]
training_4/Adam/sub_12/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
{
training_4/Adam/sub_12Subtraining_4/Adam/sub_12/x!training_4/Adam/ReadVariableOp_29*
_output_shapes
: *
T0
}
training_4/Adam/Square_3Square4training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes
:
*
T0
t
training_4/Adam/mul_19Multraining_4/Adam/sub_12training_4/Adam/Square_3*
T0*
_output_shapes
:

r
training_4/Adam/add_10Addtraining_4/Adam/mul_18training_4/Adam/mul_19*
T0*
_output_shapes
:

n
training_4/Adam/mul_20Multraining_4/Adam/multraining_4/Adam/add_9*
T0*
_output_shapes
:

\
training_4/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_4/Adam/Const_10Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Й
'training_4/Adam/clip_by_value_4/MinimumMinimumtraining_4/Adam/add_10training_4/Adam/Const_10*
_output_shapes
:
*
T0
С
training_4/Adam/clip_by_value_4Maximum'training_4/Adam/clip_by_value_4/Minimumtraining_4/Adam/Const_9*
T0*
_output_shapes
:

d
training_4/Adam/Sqrt_4Sqrttraining_4/Adam/clip_by_value_4*
_output_shapes
:
*
T0
]
training_4/Adam/add_11/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
t
training_4/Adam/add_11Addtraining_4/Adam/Sqrt_4training_4/Adam/add_11/y*
T0*
_output_shapes
:

y
training_4/Adam/truediv_4RealDivtraining_4/Adam/mul_20training_4/Adam/add_11*
T0*
_output_shapes
:

j
!training_4/Adam/ReadVariableOp_30ReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:

А
training_4/Adam/sub_13Sub!training_4/Adam/ReadVariableOp_30training_4/Adam/truediv_4*
_output_shapes
:
*
T0
v
"training_4/Adam/AssignVariableOp_9AssignVariableOptraining_4/Adam/Variable_3training_4/Adam/add_9*
dtype0
Э
!training_4/Adam/ReadVariableOp_31ReadVariableOptraining_4/Adam/Variable_3#^training_4/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:

x
#training_4/Adam/AssignVariableOp_10AssignVariableOptraining_4/Adam/Variable_7training_4/Adam/add_10*
dtype0
Ю
!training_4/Adam/ReadVariableOp_32ReadVariableOptraining_4/Adam/Variable_7$^training_4/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:

j
#training_4/Adam/AssignVariableOp_11AssignVariableOpdense_5/biastraining_4/Adam/sub_13*
dtype0
Р
!training_4/Adam/ReadVariableOp_33ReadVariableOpdense_5/bias$^training_4/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
:

Є
training_5/group_depsNoOp^loss_2/mul^metrics_2/acc/div_no_nan"^training_4/Adam/ReadVariableOp_15"^training_4/Adam/ReadVariableOp_16"^training_4/Adam/ReadVariableOp_17"^training_4/Adam/ReadVariableOp_23"^training_4/Adam/ReadVariableOp_24"^training_4/Adam/ReadVariableOp_25"^training_4/Adam/ReadVariableOp_31"^training_4/Adam/ReadVariableOp_32"^training_4/Adam/ReadVariableOp_33!^training_4/Adam/ReadVariableOp_7!^training_4/Adam/ReadVariableOp_8!^training_4/Adam/ReadVariableOp_9
N
VarIsInitializedOp_37VarIsInitializedOp	Adam_1/lr*
_output_shapes
: 
V
VarIsInitializedOp_38VarIsInitializedOpAdam_1/iterations*
_output_shapes
: 
_
VarIsInitializedOp_39VarIsInitializedOptraining_4/Adam/Variable_9*
_output_shapes
: 
S
VarIsInitializedOp_40VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
S
VarIsInitializedOp_41VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_42VarIsInitializedOpdense_4/bias*
_output_shapes
: 
`
VarIsInitializedOp_43VarIsInitializedOptraining_4/Adam/Variable_11*
_output_shapes
: 
_
VarIsInitializedOp_44VarIsInitializedOptraining_4/Adam/Variable_1*
_output_shapes
: 
L
VarIsInitializedOp_45VarIsInitializedOptotal_2*
_output_shapes
: 
R
VarIsInitializedOp_46VarIsInitializedOpAdam_1/beta_1*
_output_shapes
: 
_
VarIsInitializedOp_47VarIsInitializedOptraining_4/Adam/Variable_5*
_output_shapes
: 
`
VarIsInitializedOp_48VarIsInitializedOptraining_4/Adam/Variable_10*
_output_shapes
: 
Q
VarIsInitializedOp_49VarIsInitializedOpdense_5/bias*
_output_shapes
: 
]
VarIsInitializedOp_50VarIsInitializedOptraining_4/Adam/Variable*
_output_shapes
: 
_
VarIsInitializedOp_51VarIsInitializedOptraining_4/Adam/Variable_6*
_output_shapes
: 
_
VarIsInitializedOp_52VarIsInitializedOptraining_4/Adam/Variable_4*
_output_shapes
: 
Q
VarIsInitializedOp_53VarIsInitializedOpAdam_1/decay*
_output_shapes
: 
_
VarIsInitializedOp_54VarIsInitializedOptraining_4/Adam/Variable_7*
_output_shapes
: 
R
VarIsInitializedOp_55VarIsInitializedOpAdam_1/beta_2*
_output_shapes
: 
_
VarIsInitializedOp_56VarIsInitializedOptraining_4/Adam/Variable_8*
_output_shapes
: 
_
VarIsInitializedOp_57VarIsInitializedOptraining_4/Adam/Variable_2*
_output_shapes
: 
_
VarIsInitializedOp_58VarIsInitializedOptraining_4/Adam/Variable_3*
_output_shapes
: 
L
VarIsInitializedOp_59VarIsInitializedOpcount_2*
_output_shapes
: 
о
init_2NoOp^Adam_1/beta_1/Assign^Adam_1/beta_2/Assign^Adam_1/decay/Assign^Adam_1/iterations/Assign^Adam_1/lr/Assign^count_2/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign^total_2/Assign ^training_4/Adam/Variable/Assign"^training_4/Adam/Variable_1/Assign#^training_4/Adam/Variable_10/Assign#^training_4/Adam/Variable_11/Assign"^training_4/Adam/Variable_2/Assign"^training_4/Adam/Variable_3/Assign"^training_4/Adam/Variable_4/Assign"^training_4/Adam/Variable_5/Assign"^training_4/Adam/Variable_6/Assign"^training_4/Adam/Variable_7/Assign"^training_4/Adam/Variable_8/Assign"^training_4/Adam/Variable_9/Assign"┐L┬пз+     dЮ┌╔	KxьЮ╫AJЪ╫
ф#╨#
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
╣
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
5
DivNoNan
x"T
y"T
z"T"
Ttype:
2
B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
У
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02unknown▐║
П
)Adam/iterations/Initializer/initial_valueConst*
value	B	 R *"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
з
Adam/iterationsVarHandleOp*
_output_shapes
: * 
shared_nameAdam/iterations*
dtype0	*
shape: *
	container *"
_class
loc:@Adam/iterations
o
0Adam/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/iterations*
_output_shapes
: 
Ч
Adam/iterations/AssignAssignVariableOpAdam/iterations)Adam/iterations/Initializer/initial_value*"
_class
loc:@Adam/iterations*
dtype0	
П
#Adam/iterations/Read/ReadVariableOpReadVariableOpAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
В
!Adam/lr/Initializer/initial_valueConst*
valueB
 *oГ:*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
П
Adam/lrVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/lr*
_output_shapes
: *
shared_name	Adam/lr
_
(Adam/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/lr*
_output_shapes
: 
w
Adam/lr/AssignAssignVariableOpAdam/lr!Adam/lr/Initializer/initial_value*
_class
loc:@Adam/lr*
dtype0
w
Adam/lr/Read/ReadVariableOpReadVariableOpAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
К
%Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
Ы
Adam/beta_1VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/beta_1*
shared_nameAdam/beta_1*
_output_shapes
: 
g
,Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_1*
_output_shapes
: 
З
Adam/beta_1/AssignAssignVariableOpAdam/beta_1%Adam/beta_1/Initializer/initial_value*
_class
loc:@Adam/beta_1*
dtype0
Г
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
К
%Adam/beta_2/Initializer/initial_valueConst*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: *
valueB
 *w╛?
Ы
Adam/beta_2VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/beta_2*
_output_shapes
: *
shared_nameAdam/beta_2
g
,Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam/beta_2*
_output_shapes
: 
З
Adam/beta_2/AssignAssignVariableOpAdam/beta_2%Adam/beta_2/Initializer/initial_value*
dtype0*
_class
loc:@Adam/beta_2
Г
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
И
$Adam/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
Ш

Adam/decayVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam/decay*
shared_name
Adam/decay*
_output_shapes
: 
e
+Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Adam/decay*
_output_shapes
: 
Г
Adam/decay/AssignAssignVariableOp
Adam/decay$Adam/decay/Initializer/initial_value*
dtype0*
_class
loc:@Adam/decay
А
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
r
input_1Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
L
ShapeShapeinput_1*
_output_shapes
:*
T0*
out_type0
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
∙
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
new_axis_mask *
T0*

begin_mask *
_output_shapes
: *
shrink_axis_mask*
ellipsis_mask *
end_mask *
Index0
Z
Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
o
Reshape/shapePackstrided_sliceReshape/shape/1*
N*
T0*

axis *
_output_shapes
:
k
ReshapeReshapeinput_1Reshape/shape*
Tshape0*(
_output_shapes
:         Р*
T0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:*
valueB"     
С
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *м\▒╜*
_class
loc:@dense/kernel
С
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *м\▒=*
_class
loc:@dense/kernel
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *
T0*

seed *
_class
loc:@dense/kernel*
_output_shapes
:	Р
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Р*
T0*
_class
loc:@dense/kernel
╙
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	Р
з
dense/kernelVarHandleOp*
dtype0*
shape:	Р*
	container *
_class
loc:@dense/kernel*
shared_namedense/kernel*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
М
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0
П
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	Р
И
dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
Ь

dense/biasVarHandleOp*
dtype0*
shape:*
	container *
_class
loc:@dense/bias*
shared_name
dense/bias*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0
Д
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias
c
MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р
И
MatMulMatMulReshapeMatMul/ReadVariableOp*
transpose_b( *'
_output_shapes
:         *
T0*
transpose_a( 
]
BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
{
BiasAddBiasAddMatMulBiasAdd/ReadVariableOp*'
_output_shapes
:         *
T0*
data_formatNHWC
G
ReluReluBiasAdd*
T0*'
_output_shapes
:         
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
_output_shapes
: *
shape: *
dtype0

d
cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

I
cond/switch_tIdentitycond/Switch:1*
_output_shapes
: *
T0

G
cond/switch_fIdentitycond/Switch*
_output_shapes
: *
T0

O
cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

f
cond/dropout/rateConst^cond/switch_t*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
m
cond/dropout/ShapeShapecond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Х
cond/dropout/Shape/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
	loc:@Relu
g
cond/dropout/sub/xConst^cond/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
cond/dropout/subSubcond/dropout/sub/xcond/dropout/rate*
_output_shapes
: *
T0
t
cond/dropout/random_uniform/minConst^cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
t
cond/dropout/random_uniform/maxConst^cond/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ж
)cond/dropout/random_uniform/RandomUniformRandomUniformcond/dropout/Shape*
dtype0*'
_output_shapes
:         *
seed2 *
T0*

seed 
Й
cond/dropout/random_uniform/subSubcond/dropout/random_uniform/maxcond/dropout/random_uniform/min*
T0*
_output_shapes
: 
д
cond/dropout/random_uniform/mulMul)cond/dropout/random_uniform/RandomUniformcond/dropout/random_uniform/sub*'
_output_shapes
:         *
T0
Ц
cond/dropout/random_uniformAddcond/dropout/random_uniform/mulcond/dropout/random_uniform/min*'
_output_shapes
:         *
T0
x
cond/dropout/addAddcond/dropout/subcond/dropout/random_uniform*
T0*'
_output_shapes
:         
_
cond/dropout/FloorFloorcond/dropout/add*'
_output_shapes
:         *
T0
А
cond/dropout/truedivRealDivcond/dropout/Shape/Switch:1cond/dropout/sub*'
_output_shapes
:         *
T0
s
cond/dropout/mulMulcond/dropout/truedivcond/dropout/Floor*'
_output_shapes
:         *
T0
a
cond/IdentityIdentitycond/Identity/Switch*'
_output_shapes
:         *
T0
Р
cond/Identity/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
	loc:@Relu
q

cond/MergeMergecond/Identitycond/dropout/mul*
N*
T0*)
_output_shapes
:         : 
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї╛
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї>*!
_class
loc:@dense_1/kernel
ы
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *
T0*

seed *!
_class
loc:@dense_1/kernel*
_output_shapes

:

╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

┌
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes

:
*
T0
м
dense_1/kernelVarHandleOp*
shape
:
*
	container *!
_class
loc:@dense_1/kernel*
_output_shapes
: *
shared_namedense_1/kernel*
dtype0
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
Ф
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0
Ф
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:
*!
_class
loc:@dense_1/kernel
М
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
*
valueB
*    
в
dense_1/biasVarHandleOp*
shape:
*
	container *
_class
loc:@dense_1/bias*
_output_shapes
: *
shared_namedense_1/bias*
dtype0
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
Г
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0
К
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:

f
MatMul_1/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

П
MatMul_1MatMul
cond/MergeMatMul_1/ReadVariableOp*
transpose_b( *'
_output_shapes
:         
*
T0*
transpose_a( 
a
BiasAdd_1/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_1BiasAddMatMul_1BiasAdd_1/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:         
*
T0
O
SoftmaxSoftmax	BiasAdd_1*
T0*'
_output_shapes
:         

Д
output_1_targetPlaceholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
R
ConstConst*
valueB*  А?*
dtype0*
_output_shapes
:
Д
output_1_sample_weightsPlaceholderWithDefaultConst*
shape:         *
dtype0*#
_output_shapes
:         
v
total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class

loc:@total
Й
totalVarHandleOp*
_class

loc:@total*
_output_shapes
: *
shared_nametotal*
dtype0*
shape: *
	container 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
_class

loc:@total*
dtype0
v
count/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
_class

loc:@count*
dtype0
Й
countVarHandleOp*
dtype0*
shape: *
	container *
_class

loc:@count*
_output_shapes
: *
shared_namecount
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: *
_class

loc:@count
s
 loss/output_1_loss/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Ф
loss/output_1_loss/ReshapeReshapeoutput_1_target loss/output_1_loss/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
И
loss/output_1_loss/CastCastloss/output_1_loss/Reshape*

DstT0	*#
_output_shapes
:         *

SrcT0*
Truncate( 
s
"loss/output_1_loss/Reshape_1/shapeConst*
valueB"    
   *
dtype0*
_output_shapes
:
Ц
loss/output_1_loss/Reshape_1Reshape	BiasAdd_1"loss/output_1_loss/Reshape_1/shape*
Tshape0*'
_output_shapes
:         
*
T0
У
<loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/output_1_loss/Cast*
_output_shapes
:*
T0	*
out_type0
И
Zloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/output_1_loss/Reshape_1loss/output_1_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":         :         

Ю
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights*
out_type0*
_output_shapes
:*
T0
И
Floss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
р
Floss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
З
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
№
Closs/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualEloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Ж
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

╤
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentityQloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

╧
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityOloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

┬
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

э
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0
*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar
Л
oloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualvloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchxloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
Ц
vloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchEloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*X
_classN
LJloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
Ъ
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankPloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : *
T0
°
iloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

Е
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitykloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Г
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityiloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
И
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityoloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

╝
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
valueB :
         
╥
~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЙloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
░
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
М
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЕloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
├
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
┤
Гloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╠
}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeГloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
п
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B :
─
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2~loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims}loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
_output_shapes

:*
N*
T0
╛
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЛloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*
T0*

Tdim0
┤
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapePloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
С
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЗloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Я
Мloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationАloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
set_operationa-b*<
_output_shapes*
(:         :         :*
validate_indices(
╧
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeОloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
out_type0*
_output_shapes
: *
T0
е
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstl^loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
Ы
sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualuloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xДloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
·
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switcholoss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankjloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*В
_classx
vtloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

 
hloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergekloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
┬
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergehloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeSloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
N*
T0

з
?loss/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Р
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
_output_shapes
: *
valueB Bweights.shape=*
dtype0
Ы
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*
dtype0*
_output_shapes
: **
value!B Boutput_1_sample_weights:0
П
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
▐
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*
_output_shapes
: *m
valuedBb B\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0
М
Aloss/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Щ
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

╦
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

╔
Nloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityLloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
_output_shapes
: *
T0

╩
Mloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

г
Jloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Е
Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tK^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: *
T0

М
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
є
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bweights.shape=
■
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: **
value!B Boutput_1_sample_weights:0
Є
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
┴
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *m
valuedBb B\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0
я
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstO^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╙
Lloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
В
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchNloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
■
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchGloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*Z
_classP
NLloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
№
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchFloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*Y
_classO
MKloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
ю
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchCloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarMloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*V
_classL
JHloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

Й
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityNloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fM^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*a
_classW
USloc:@loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
T0

╢
Kloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergeZloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Xloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
N*
T0

Ь
4loss/output_1_loss/broadcast_weights/ones_like/ShapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
╟
4loss/output_1_loss/broadcast_weights/ones_like/ConstConstL^loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
т
.loss/output_1_loss/broadcast_weights/ones_likeFill4loss/output_1_loss/broadcast_weights/ones_like/Shape4loss/output_1_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
в
$loss/output_1_loss/broadcast_weightsMuloutput_1_sample_weights.loss/output_1_loss/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
═
loss/output_1_loss/MulMulZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits$loss/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0
b
loss/output_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Н
loss/output_1_loss/SumSumloss/output_1_loss/Mulloss/output_1_loss/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
d
loss/output_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Я
loss/output_1_loss/Sum_1Sum$loss/output_1_loss/broadcast_weightsloss/output_1_loss/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
|
loss/output_1_loss/div_no_nanDivNoNanloss/output_1_loss/Sumloss/output_1_loss/Sum_1*
_output_shapes
: *
T0
]
loss/output_1_loss/Const_2Const*
_output_shapes
: *
valueB *
dtype0
Ш
loss/output_1_loss/MeanMeanloss/output_1_loss/div_no_nanloss/output_1_loss/Const_2*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/output_1_loss/Mean*
_output_shapes
: *
T0
Г
metrics/acc/CastCastoutput_1_target*0
_output_shapes
:                  *

SrcT0*
Truncate( *

DstT0
~
metrics/acc/SqueezeSqueezemetrics/acc/Cast*#
_output_shapes
:         *
T0*
squeeze_dims

         
g
metrics/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Р
metrics/acc/ArgMaxArgMaxSoftmaxmetrics/acc/ArgMax/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:         
{
metrics/acc/Cast_1Castmetrics/acc/ArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:         
q
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast_1*#
_output_shapes
:         *
T0
z
metrics/acc/Cast_2Castmetrics/acc/Equal*

DstT0*#
_output_shapes
:         *

SrcT0
*
Truncate( 
]
metrics/acc/SizeSizemetrics/acc/Cast_2*
_output_shapes
: *
T0*
out_type0
l
metrics/acc/Cast_3Castmetrics/acc/Size*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/SumSummetrics/acc/Cast_2metrics/acc/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
z
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
}
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_3^metrics/acc/ReadVariableOp*
dtype0
Ы
metrics/acc/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1^metrics/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
В
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal^metrics/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Д
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount^metrics/acc/ReadVariableOp_1*
_output_shapes
: *
dtype0
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0

metrics/acc/Squeeze_1Squeezeoutput_1_target*#
_output_shapes
:         *
T0*
squeeze_dims

         
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ф
metrics/acc/ArgMax_1ArgMaxSoftmaxmetrics/acc/ArgMax_1/dimension*
output_type0	*

Tidx0*#
_output_shapes
:         *
T0
}
metrics/acc/Cast_4Castmetrics/acc/ArgMax_1*

DstT0*#
_output_shapes
:         *

SrcT0	*
Truncate( 
u
metrics/acc/Equal_1Equalmetrics/acc/Squeeze_1metrics/acc/Cast_4*
T0*#
_output_shapes
:         
|
metrics/acc/Cast_5Castmetrics/acc/Equal_1*#
_output_shapes
:         *

SrcT0
*
Truncate( *

DstT0
]
metrics/acc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

metrics/acc/MeanMeanmetrics/acc/Cast_5metrics/acc/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
dtype0*
_output_shapes
: *
valueB 
Г
!training/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  А?*
_class
loc:@loss/mul*
dtype0
╢
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*

index_type0*
_class
loc:@loss/mul*
_output_shapes
: *
T0
е
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/output_1_loss/Mean*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ъ
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
▒
Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB **
_class 
loc:@loss/output_1_loss/Mean*
dtype0*
_output_shapes
: 
У
<training/Adam/gradients/loss/output_1_loss/Mean_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Btraining/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape/shape*
_output_shapes
: *
T0**
_class 
loc:@loss/output_1_loss/Mean*
Tshape0
й
:training/Adam/gradients/loss/output_1_loss/Mean_grad/ConstConst*
valueB **
_class 
loc:@loss/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ъ
9training/Adam/gradients/loss/output_1_loss/Mean_grad/TileTile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Reshape:training/Adam/gradients/loss/output_1_loss/Mean_grad/Const*
_output_shapes
: *
T0*

Tmultiples0**
_class 
loc:@loss/output_1_loss/Mean
н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1Const*
valueB
 *  А?**
_class 
loc:@loss/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Н
<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivRealDiv9training/Adam/gradients/loss/output_1_loss/Mean_grad/Tile<training/Adam/gradients/loss/output_1_loss/Mean_grad/Const_1*
T0**
_class 
loc:@loss/output_1_loss/Mean*
_output_shapes
: 
╡
@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ShapeConst*
_output_shapes
: *
valueB *0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
dtype0
╖
Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *0
_class&
$"loc:@loss/output_1_loss/div_no_nan
▐
Ptraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgs@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ShapeBtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1*2
_output_shapes 
:         :         *
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan
№
Etraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivloss/output_1_loss/Sum_1*
_output_shapes
: *
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan
╬
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/SumSumEtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nanPtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
	keep_dims( *

Tidx0*
_output_shapes
: 
░
Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ReshapeReshape>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: 
░
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/NegNegloss/output_1_loss/Sum*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: 
А
Gtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNan>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Negloss/output_1_loss/Sum_1*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
Й
Gtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanGtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_1loss/output_1_loss/Sum_1*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: 
Я
>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/mulMul<training/Adam/gradients/loss/output_1_loss/Mean_grad/truedivGtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/div_no_nan_2*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╦
@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum_1Sum>training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/mulRtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
	keep_dims( *

Tidx0*
_output_shapes
: 
╢
Dtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Reshape_1Reshape@training/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Sum_1Btraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/Shape_1*
T0*0
_class&
$"loc:@loss/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: 
╢
Atraining/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*)
_class
loc:@loss/output_1_loss/Sum*
dtype0*
_output_shapes
:
л
;training/Adam/gradients/loss/output_1_loss/Sum_grad/ReshapeReshapeBtraining/Adam/gradients/loss/output_1_loss/div_no_nan_grad/ReshapeAtraining/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape/shape*
T0*)
_class
loc:@loss/output_1_loss/Sum*
Tshape0*
_output_shapes
:
║
9training/Adam/gradients/loss/output_1_loss/Sum_grad/ShapeShapeloss/output_1_loss/Mul*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/output_1_loss/Sum
г
8training/Adam/gradients/loss/output_1_loss/Sum_grad/TileTile;training/Adam/gradients/loss/output_1_loss/Sum_grad/Reshape9training/Adam/gradients/loss/output_1_loss/Sum_grad/Shape*
T0*

Tmultiples0*)
_class
loc:@loss/output_1_loss/Sum*#
_output_shapes
:         
■
9training/Adam/gradients/loss/output_1_loss/Mul_grad/ShapeShapeZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/output_1_loss/Mul
╩
;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1Shape$loss/output_1_loss/broadcast_weights*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/output_1_loss/Mul
┬
Itraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs9training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1*
T0*)
_class
loc:@loss/output_1_loss/Mul*2
_output_shapes 
:         :         
ў
7training/Adam/gradients/loss/output_1_loss/Mul_grad/MulMul8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile$loss/output_1_loss/broadcast_weights*
T0*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         
н
7training/Adam/gradients/loss/output_1_loss/Mul_grad/SumSum7training/Adam/gradients/loss/output_1_loss/Mul_grad/MulItraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgs*)
_class
loc:@loss/output_1_loss/Mul*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
б
;training/Adam/gradients/loss/output_1_loss/Mul_grad/ReshapeReshape7training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum9training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape*#
_output_shapes
:         *
T0*)
_class
loc:@loss/output_1_loss/Mul*
Tshape0
п
9training/Adam/gradients/loss/output_1_loss/Mul_grad/Mul_1MulZloss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits8training/Adam/gradients/loss/output_1_loss/Sum_grad/Tile*
T0*)
_class
loc:@loss/output_1_loss/Mul*#
_output_shapes
:         
│
9training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum_1Sum9training/Adam/gradients/loss/output_1_loss/Mul_grad/Mul_1Ktraining/Adam/gradients/loss/output_1_loss/Mul_grad/BroadcastGradientArgs:1*
T0*)
_class
loc:@loss/output_1_loss/Mul*
	keep_dims( *

Tidx0*
_output_shapes
:
з
=training/Adam/gradients/loss/output_1_loss/Mul_grad/Reshape_1Reshape9training/Adam/gradients/loss/output_1_loss/Mul_grad/Sum_1;training/Adam/gradients/loss/output_1_loss/Mul_grad/Shape_1*)
_class
loc:@loss/output_1_loss/Mul*
Tshape0*#
_output_shapes
:         *
T0
о
"training/Adam/gradients/zeros_like	ZerosLike\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╤
Зtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient\loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

┴
Жtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0
Д
Вtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/output_1_loss/Mul_grad/ReshapeЖtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
▓
{training/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulВtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsЗtraining/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*m
_classc
a_loc:@loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
*
T0
╣
?training/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_1*/
_class%
#!loc:@loss/output_1_loss/Reshape_1*
_output_shapes
:*
T0*
out_type0
√
Atraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ReshapeReshape{training/Adam/gradients/loss/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul?training/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Shape*/
_class%
#!loc:@loss/output_1_loss/Reshape_1*
Tshape0*'
_output_shapes
:         
*
T0
▐
2training/Adam/gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Reshape*
data_formatNHWC*
_class
loc:@BiasAdd_1*
_output_shapes
:
*
T0
З
,training/Adam/gradients/MatMul_1_grad/MatMulMatMulAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/ReshapeMatMul_1/ReadVariableOp*
transpose_a( *
T0*
_class
loc:@MatMul_1*
transpose_b(*'
_output_shapes
:         
є
.training/Adam/gradients/MatMul_1_grad/MatMul_1MatMul
cond/MergeAtraining/Adam/gradients/loss/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@MatMul_1*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
┘
1training/Adam/gradients/cond/Merge_grad/cond_gradSwitch,training/Adam/gradients/MatMul_1_grad/MatMulcond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@MatMul_1
м
3training/Adam/gradients/cond/dropout/mul_grad/ShapeShapecond/dropout/truediv*
T0*
out_type0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
м
5training/Adam/gradients/cond/dropout/mul_grad/Shape_1Shapecond/dropout/Floor*
T0*
out_type0*#
_class
loc:@cond/dropout/mul*
_output_shapes
:
к
Ctraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs3training/Adam/gradients/cond/dropout/mul_grad/Shape5training/Adam/gradients/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0*#
_class
loc:@cond/dropout/mul
╪
1training/Adam/gradients/cond/dropout/mul_grad/MulMul3training/Adam/gradients/cond/Merge_grad/cond_grad:1cond/dropout/Floor*
T0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:         
Х
1training/Adam/gradients/cond/dropout/mul_grad/SumSum1training/Adam/gradients/cond/dropout/mul_grad/MulCtraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs*
T0*#
_class
loc:@cond/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:
Н
5training/Adam/gradients/cond/dropout/mul_grad/ReshapeReshape1training/Adam/gradients/cond/dropout/mul_grad/Sum3training/Adam/gradients/cond/dropout/mul_grad/Shape*'
_output_shapes
:         *
T0*#
_class
loc:@cond/dropout/mul*
Tshape0
▄
3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Mulcond/dropout/truediv3training/Adam/gradients/cond/Merge_grad/cond_grad:1*
T0*#
_class
loc:@cond/dropout/mul*'
_output_shapes
:         
Ы
3training/Adam/gradients/cond/dropout/mul_grad/Sum_1Sum3training/Adam/gradients/cond/dropout/mul_grad/Mul_1Etraining/Adam/gradients/cond/dropout/mul_grad/BroadcastGradientArgs:1*
T0*#
_class
loc:@cond/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:
У
7training/Adam/gradients/cond/dropout/mul_grad/Reshape_1Reshape3training/Adam/gradients/cond/dropout/mul_grad/Sum_15training/Adam/gradients/cond/dropout/mul_grad/Shape_1*'
_output_shapes
:         *
T0*#
_class
loc:@cond/dropout/mul*
Tshape0
Ъ
training/Adam/gradients/SwitchSwitchRelucond/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
	loc:@Relu
Щ
 training/Adam/gradients/IdentityIdentity training/Adam/gradients/Switch:1*'
_output_shapes
:         *
T0*
_class
	loc:@Relu
Ш
training/Adam/gradients/Shape_1Shape training/Adam/gradients/Switch:1*
T0*
out_type0*
_class
	loc:@Relu*
_output_shapes
:
д
#training/Adam/gradients/zeros/ConstConst!^training/Adam/gradients/Identity*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
	loc:@Relu
╚
training/Adam/gradients/zerosFilltraining/Adam/gradients/Shape_1#training/Adam/gradients/zeros/Const*
T0*

index_type0*
_class
	loc:@Relu*'
_output_shapes
:         
ь
;training/Adam/gradients/cond/Identity/Switch_grad/cond_gradMerge1training/Adam/gradients/cond/Merge_grad/cond_gradtraining/Adam/gradients/zeros*
T0*
N*
_class
	loc:@Relu*)
_output_shapes
:         : 
╗
7training/Adam/gradients/cond/dropout/truediv_grad/ShapeShapecond/dropout/Shape/Switch:1*
T0*
out_type0*'
_class
loc:@cond/dropout/truediv*
_output_shapes
:
е
9training/Adam/gradients/cond/dropout/truediv_grad/Shape_1Const*
valueB *'
_class
loc:@cond/dropout/truediv*
dtype0*
_output_shapes
: 
║
Gtraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs7training/Adam/gradients/cond/dropout/truediv_grad/Shape9training/Adam/gradients/cond/dropout/truediv_grad/Shape_1*'
_class
loc:@cond/dropout/truediv*2
_output_shapes 
:         :         *
T0
ш
9training/Adam/gradients/cond/dropout/truediv_grad/RealDivRealDiv5training/Adam/gradients/cond/dropout/mul_grad/Reshapecond/dropout/sub*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         *
T0
й
5training/Adam/gradients/cond/dropout/truediv_grad/SumSum9training/Adam/gradients/cond/dropout/truediv_grad/RealDivGtraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgs*'
_class
loc:@cond/dropout/truediv*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Э
9training/Adam/gradients/cond/dropout/truediv_grad/ReshapeReshape5training/Adam/gradients/cond/dropout/truediv_grad/Sum7training/Adam/gradients/cond/dropout/truediv_grad/Shape*
T0*'
_class
loc:@cond/dropout/truediv*
Tshape0*'
_output_shapes
:         
┤
5training/Adam/gradients/cond/dropout/truediv_grad/NegNegcond/dropout/Shape/Switch:1*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
ъ
;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_1RealDiv5training/Adam/gradients/cond/dropout/truediv_grad/Negcond/dropout/sub*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
Ё
;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_2RealDiv;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_1cond/dropout/sub*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
Л
5training/Adam/gradients/cond/dropout/truediv_grad/mulMul5training/Adam/gradients/cond/dropout/mul_grad/Reshape;training/Adam/gradients/cond/dropout/truediv_grad/RealDiv_2*
T0*'
_class
loc:@cond/dropout/truediv*'
_output_shapes
:         
й
7training/Adam/gradients/cond/dropout/truediv_grad/Sum_1Sum5training/Adam/gradients/cond/dropout/truediv_grad/mulItraining/Adam/gradients/cond/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*'
_class
loc:@cond/dropout/truediv*
	keep_dims( *

Tidx0*
_output_shapes
:
Т
;training/Adam/gradients/cond/dropout/truediv_grad/Reshape_1Reshape7training/Adam/gradients/cond/dropout/truediv_grad/Sum_19training/Adam/gradients/cond/dropout/truediv_grad/Shape_1*'
_class
loc:@cond/dropout/truediv*
Tshape0*
_output_shapes
: *
T0
Ь
 training/Adam/gradients/Switch_1SwitchRelucond/pred_id*
_class
	loc:@Relu*:
_output_shapes(
&:         :         *
T0
Ы
"training/Adam/gradients/Identity_1Identity training/Adam/gradients/Switch_1*
T0*
_class
	loc:@Relu*'
_output_shapes
:         
Ш
training/Adam/gradients/Shape_2Shape training/Adam/gradients/Switch_1*
_output_shapes
:*
T0*
out_type0*
_class
	loc:@Relu
и
%training/Adam/gradients/zeros_1/ConstConst#^training/Adam/gradients/Identity_1*
valueB
 *    *
_class
	loc:@Relu*
dtype0*
_output_shapes
: 
╠
training/Adam/gradients/zeros_1Filltraining/Adam/gradients/Shape_2%training/Adam/gradients/zeros_1/Const*
_class
	loc:@Relu*'
_output_shapes
:         *
T0*

index_type0
√
@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_gradMergetraining/Adam/gradients/zeros_19training/Adam/gradients/cond/dropout/truediv_grad/Reshape*
_class
	loc:@Relu*)
_output_shapes
:         : *
T0*
N
ў
training/Adam/gradients/AddNAddN;training/Adam/gradients/cond/Identity/Switch_grad/cond_grad@training/Adam/gradients/cond/dropout/Shape/Switch_grad/cond_grad*'
_output_shapes
:         *
T0*
N*
_class
	loc:@Relu
е
*training/Adam/gradients/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddNRelu*
_class
	loc:@Relu*'
_output_shapes
:         *
T0
├
0training/Adam/gradients/BiasAdd_grad/BiasAddGradBiasAddGrad*training/Adam/gradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_class
loc:@BiasAdd*
_output_shapes
:
ы
*training/Adam/gradients/MatMul_grad/MatMulMatMul*training/Adam/gradients/Relu_grad/ReluGradMatMul/ReadVariableOp*
transpose_a( *
T0*
_class
loc:@MatMul*
transpose_b(*(
_output_shapes
:         Р
╓
,training/Adam/gradients/MatMul_grad/MatMul_1MatMulReshape*training/Adam/gradients/Relu_grad/ReluGrad*
transpose_a(*
T0*
_class
loc:@MatMul*
transpose_b( *
_output_shapes
:	Р
U
training/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
k
!training/Adam/AssignAddVariableOpAssignAddVariableOpAdam/iterationstraining/Adam/Const*
dtype0	
И
training/Adam/ReadVariableOpReadVariableOpAdam/iterations"^training/Adam/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
И
!training/Adam/Cast/ReadVariableOpReadVariableOpAdam/iterations^training/Adam/ReadVariableOp*
dtype0	*
_output_shapes
: 
}
training/Adam/CastCast!training/Adam/Cast/ReadVariableOp*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
d
 training/Adam/Pow/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
o
training/Adam/PowPow training/Adam/Pow/ReadVariableOptraining/Adam/Cast*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
Z
training/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_2Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_2*
_output_shapes
: *
T0
Г
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const_1*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
f
"training/Adam/Pow_1/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
s
training/Adam/Pow_1Pow"training/Adam/Pow_1/ReadVariableOptraining/Adam/Cast*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/ReadVariableOp_1ReadVariableOpAdam/lr*
dtype0*
_output_shapes
: 
p
training/Adam/mulMultraining/Adam/ReadVariableOp_1training/Adam/truediv*
_output_shapes
: *
T0
t
#training/Adam/zeros/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
_output_shapes
:	Р*
T0*

index_type0
┼
training/Adam/VariableVarHandleOp*
dtype0*
shape:	Р*
	container *)
_class
loc:@training/Adam/Variable*'
shared_nametraining/Adam/Variable*
_output_shapes
: 
}
7training/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
Ц
training/Adam/Variable/AssignAssignVariableOptraining/Adam/Variabletraining/Adam/zeros*)
_class
loc:@training/Adam/Variable*
dtype0
н
*training/Adam/Variable/Read/ReadVariableOpReadVariableOptraining/Adam/Variable*
dtype0*
_output_shapes
:	Р*)
_class
loc:@training/Adam/Variable
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
╞
training/Adam/Variable_1VarHandleOp*
dtype0*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: *)
shared_nametraining/Adam/Variable_1
Б
9training/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
Ю
training/Adam/Variable_1/AssignAssignVariableOptraining/Adam/Variable_1training/Adam/zeros_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0
о
,training/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
:
j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

╩
training/Adam/Variable_2VarHandleOp*)
shared_nametraining/Adam/Variable_2*
_output_shapes
: *
dtype0*
shape
:
*
	container *+
_class!
loc:@training/Adam/Variable_2
Б
9training/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
Ю
training/Adam/Variable_2/AssignAssignVariableOptraining/Adam/Variable_2training/Adam/zeros_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0
▓
,training/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes

:

b
training/Adam/zeros_3Const*
valueB
*    *
dtype0*
_output_shapes
:

╞
training/Adam/Variable_3VarHandleOp*
shape:
*
	container *+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: *)
shared_nametraining/Adam/Variable_3*
dtype0
Б
9training/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
Ю
training/Adam/Variable_3/AssignAssignVariableOptraining/Adam/Variable_3training/Adam/zeros_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0
о
,training/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_3*
dtype0
v
%training/Adam/zeros_4/shape_as_tensorConst*
_output_shapes
:*
valueB"     *
dtype0
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes
:	Р
╦
training/Adam/Variable_4VarHandleOp*
shape:	Р*
	container *+
_class!
loc:@training/Adam/Variable_4*)
shared_nametraining/Adam/Variable_4*
_output_shapes
: *
dtype0
Б
9training/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
Ю
training/Adam/Variable_4/AssignAssignVariableOptraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0
│
,training/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
╞
training/Adam/Variable_5VarHandleOp*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: *)
shared_nametraining/Adam/Variable_5*
dtype0
Б
9training/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
Ю
training/Adam/Variable_5/AssignAssignVariableOptraining/Adam/Variable_5training/Adam/zeros_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0
о
,training/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
dtype0*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_5
j
training/Adam/zeros_6Const*
_output_shapes

:
*
valueB
*    *
dtype0
╩
training/Adam/Variable_6VarHandleOp*
shape
:
*
	container *+
_class!
loc:@training/Adam/Variable_6*)
shared_nametraining/Adam/Variable_6*
_output_shapes
: *
dtype0
Б
9training/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
Ю
training/Adam/Variable_6/AssignAssignVariableOptraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0
▓
,training/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes

:

b
training/Adam/zeros_7Const*
valueB
*    *
dtype0*
_output_shapes
:

╞
training/Adam/Variable_7VarHandleOp*)
shared_nametraining/Adam/Variable_7*
_output_shapes
: *
dtype0*
shape:
*
	container *+
_class!
loc:@training/Adam/Variable_7
Б
9training/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
Ю
training/Adam/Variable_7/AssignAssignVariableOptraining/Adam/Variable_7training/Adam/zeros_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0
о
,training/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_7
o
%training/Adam/zeros_8/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*
_output_shapes
:
╞
training/Adam/Variable_8VarHandleOp*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: *)
shared_nametraining/Adam/Variable_8*
dtype0
Б
9training/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
Ю
training/Adam/Variable_8/AssignAssignVariableOptraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0
о
,training/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_8*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_8*
dtype0
o
%training/Adam/zeros_9/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
`
training/Adam/zeros_9/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ш
training/Adam/zeros_9Fill%training/Adam/zeros_9/shape_as_tensortraining/Adam/zeros_9/Const*
_output_shapes
:*
T0*

index_type0
╞
training/Adam/Variable_9VarHandleOp*
dtype0*
shape:*
	container *+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
: *)
shared_nametraining/Adam/Variable_9
Б
9training/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
Ю
training/Adam/Variable_9/AssignAssignVariableOptraining/Adam/Variable_9training/Adam/zeros_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0
о
,training/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_10/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*

index_type0*
_output_shapes
:*
T0
╔
training/Adam/Variable_10VarHandleOp*
dtype0*
shape:*
	container *,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes
: **
shared_nametraining/Adam/Variable_10
Г
:training/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
в
 training/Adam/Variable_10/AssignAssignVariableOptraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0
▒
-training/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
:
p
&training/Adam/zeros_11/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_11Fill&training/Adam/zeros_11/shape_as_tensortraining/Adam/zeros_11/Const*
T0*

index_type0*
_output_shapes
:
╔
training/Adam/Variable_11VarHandleOp*
_output_shapes
: **
shared_nametraining/Adam/Variable_11*
dtype0*
shape:*
	container *,
_class"
 loc:@training/Adam/Variable_11
Г
:training/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
в
 training/Adam/Variable_11/AssignAssignVariableOptraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0
▒
-training/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
:
b
training/Adam/ReadVariableOp_2ReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
z
"training/Adam/mul_1/ReadVariableOpReadVariableOptraining/Adam/Variable*
_output_shapes
:	Р*
dtype0
И
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2"training/Adam/mul_1/ReadVariableOp*
_output_shapes
:	Р*
T0
b
training/Adam/ReadVariableOp_3ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/ReadVariableOp_3*
_output_shapes
: *
T0
З
training/Adam/mul_2Multraining/Adam/sub_2,training/Adam/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	Р*
T0
l
training/Adam/addAddtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes
:	Р*
T0
b
training/Adam/ReadVariableOp_4ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
|
"training/Adam/mul_3/ReadVariableOpReadVariableOptraining/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
И
training/Adam/mul_3Multraining/Adam/ReadVariableOp_4"training/Adam/mul_3/ReadVariableOp*
_output_shapes
:	Р*
T0
b
training/Adam/ReadVariableOp_5ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
r
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 
v
training/Adam/SquareSquare,training/Adam/gradients/MatMul_grad/MatMul_1*
_output_shapes
:	Р*
T0
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes
:	Р*
T0
n
training/Adam/add_1Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes
:	Р
j
training/Adam/mul_5Multraining/Adam/multraining/Adam/add*
_output_shapes
:	Р*
T0
Z
training/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Ж
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_1training/Adam/Const_4*
T0*
_output_shapes
:	Р
Р
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_3*
_output_shapes
:	Р*
T0
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes
:	Р
Z
training/Adam/add_2/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
q
training/Adam/add_2Addtraining/Adam/Sqrt_1training/Adam/add_2/y*
_output_shapes
:	Р*
T0
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_2*
_output_shapes
:	Р*
T0
l
training/Adam/ReadVariableOp_6ReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	Р
}
training/Adam/sub_4Subtraining/Adam/ReadVariableOp_6training/Adam/truediv_1*
_output_shapes
:	Р*
T0
j
training/Adam/AssignVariableOpAssignVariableOptraining/Adam/Variabletraining/Adam/add*
dtype0
Ч
training/Adam/ReadVariableOp_7ReadVariableOptraining/Adam/Variable^training/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Р
p
 training/Adam/AssignVariableOp_1AssignVariableOptraining/Adam/Variable_4training/Adam/add_1*
dtype0
Ы
training/Adam/ReadVariableOp_8ReadVariableOptraining/Adam/Variable_4!^training/Adam/AssignVariableOp_1*
_output_shapes
:	Р*
dtype0
d
 training/Adam/AssignVariableOp_2AssignVariableOpdense/kerneltraining/Adam/sub_4*
dtype0
П
training/Adam/ReadVariableOp_9ReadVariableOpdense/kernel!^training/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Р
c
training/Adam/ReadVariableOp_10ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_6/ReadVariableOpReadVariableOptraining/Adam/Variable_1*
_output_shapes
:*
dtype0
Д
training/Adam/mul_6Multraining/Adam/ReadVariableOp_10"training/Adam/mul_6/ReadVariableOp*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_11ReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
Z
training/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_5Subtraining/Adam/sub_5/xtraining/Adam/ReadVariableOp_11*
T0*
_output_shapes
: 
Ж
training/Adam/mul_7Multraining/Adam/sub_50training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
i
training/Adam/add_3Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:*
T0
c
training/Adam/ReadVariableOp_12ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
w
"training/Adam/mul_8/ReadVariableOpReadVariableOptraining/Adam/Variable_5*
_output_shapes
:*
dtype0
Д
training/Adam/mul_8Multraining/Adam/ReadVariableOp_12"training/Adam/mul_8/ReadVariableOp*
T0*
_output_shapes
:
c
training/Adam/ReadVariableOp_13ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_6Subtraining/Adam/sub_6/xtraining/Adam/ReadVariableOp_13*
T0*
_output_shapes
: 
w
training/Adam/Square_1Square0training/Adam/gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_4Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_3*
_output_shapes
:*
T0
Z
training/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_6Const*
_output_shapes
: *
valueB
 *  А*
dtype0
Б
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_4training/Adam/Const_6*
T0*
_output_shapes
:
Л
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_5*
_output_shapes
:*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:*
T0
Z
training/Adam/add_5/yConst*
dtype0*
_output_shapes
: *
valueB
 *Х┐╓3
l
training/Adam/add_5Addtraining/Adam/Sqrt_2training/Adam/add_5/y*
_output_shapes
:*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_5*
T0*
_output_shapes
:
f
training/Adam/ReadVariableOp_14ReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
y
training/Adam/sub_7Subtraining/Adam/ReadVariableOp_14training/Adam/truediv_2*
_output_shapes
:*
T0
p
 training/Adam/AssignVariableOp_3AssignVariableOptraining/Adam/Variable_1training/Adam/add_3*
dtype0
Ч
training/Adam/ReadVariableOp_15ReadVariableOptraining/Adam/Variable_1!^training/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:
p
 training/Adam/AssignVariableOp_4AssignVariableOptraining/Adam/Variable_5training/Adam/add_4*
dtype0
Ч
training/Adam/ReadVariableOp_16ReadVariableOptraining/Adam/Variable_5!^training/Adam/AssignVariableOp_4*
dtype0*
_output_shapes
:
b
 training/Adam/AssignVariableOp_5AssignVariableOp
dense/biastraining/Adam/sub_7*
dtype0
Й
training/Adam/ReadVariableOp_17ReadVariableOp
dense/bias!^training/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:
c
training/Adam/ReadVariableOp_18ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_11/ReadVariableOpReadVariableOptraining/Adam/Variable_2*
_output_shapes

:
*
dtype0
К
training/Adam/mul_11Multraining/Adam/ReadVariableOp_18#training/Adam/mul_11/ReadVariableOp*
_output_shapes

:
*
T0
c
training/Adam/ReadVariableOp_19ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_8Subtraining/Adam/sub_8/xtraining/Adam/ReadVariableOp_19*
_output_shapes
: *
T0
Й
training/Adam/mul_12Multraining/Adam/sub_8.training/Adam/gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_6Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:

c
training/Adam/ReadVariableOp_20ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
|
#training/Adam/mul_13/ReadVariableOpReadVariableOptraining/Adam/Variable_6*
dtype0*
_output_shapes

:

К
training/Adam/mul_13Multraining/Adam/ReadVariableOp_20#training/Adam/mul_13/ReadVariableOp*
_output_shapes

:
*
T0
c
training/Adam/ReadVariableOp_21ReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
Z
training/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
s
training/Adam/sub_9Subtraining/Adam/sub_9/xtraining/Adam/ReadVariableOp_21*
T0*
_output_shapes
: 
y
training/Adam/Square_2Square.training/Adam/gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
*
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_7Addtraining/Adam/mul_13training/Adam/mul_14*
_output_shapes

:
*
T0
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_6*
_output_shapes

:
*
T0
Z
training/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Е
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_7training/Adam/Const_8*
T0*
_output_shapes

:

П
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_7*
_output_shapes

:
*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
_output_shapes

:
*
T0
Z
training/Adam/add_8/yConst*
_output_shapes
: *
valueB
 *Х┐╓3*
dtype0
p
training/Adam/add_8Addtraining/Adam/Sqrt_3training/Adam/add_8/y*
_output_shapes

:
*
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_8*
T0*
_output_shapes

:

n
training/Adam/ReadVariableOp_22ReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

~
training/Adam/sub_10Subtraining/Adam/ReadVariableOp_22training/Adam/truediv_3*
T0*
_output_shapes

:

p
 training/Adam/AssignVariableOp_6AssignVariableOptraining/Adam/Variable_2training/Adam/add_6*
dtype0
Ы
training/Adam/ReadVariableOp_23ReadVariableOptraining/Adam/Variable_2!^training/Adam/AssignVariableOp_6*
_output_shapes

:
*
dtype0
p
 training/Adam/AssignVariableOp_7AssignVariableOptraining/Adam/Variable_6training/Adam/add_7*
dtype0
Ы
training/Adam/ReadVariableOp_24ReadVariableOptraining/Adam/Variable_6!^training/Adam/AssignVariableOp_7*
dtype0*
_output_shapes

:

g
 training/Adam/AssignVariableOp_8AssignVariableOpdense_1/kerneltraining/Adam/sub_10*
dtype0
С
training/Adam/ReadVariableOp_25ReadVariableOpdense_1/kernel!^training/Adam/AssignVariableOp_8*
dtype0*
_output_shapes

:

c
training/Adam/ReadVariableOp_26ReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_16/ReadVariableOpReadVariableOptraining/Adam/Variable_3*
_output_shapes
:
*
dtype0
Ж
training/Adam/mul_16Multraining/Adam/ReadVariableOp_26#training/Adam/mul_16/ReadVariableOp*
_output_shapes
:
*
T0
c
training/Adam/ReadVariableOp_27ReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
[
training/Adam/sub_11/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
u
training/Adam/sub_11Subtraining/Adam/sub_11/xtraining/Adam/ReadVariableOp_27*
_output_shapes
: *
T0
К
training/Adam/mul_17Multraining/Adam/sub_112training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
*
T0
k
training/Adam/add_9Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
:
*
T0
c
training/Adam/ReadVariableOp_28ReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
x
#training/Adam/mul_18/ReadVariableOpReadVariableOptraining/Adam/Variable_7*
dtype0*
_output_shapes
:

Ж
training/Adam/mul_18Multraining/Adam/ReadVariableOp_28#training/Adam/mul_18/ReadVariableOp*
_output_shapes
:
*
T0
c
training/Adam/ReadVariableOp_29ReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
u
training/Adam/sub_12Subtraining/Adam/sub_12/xtraining/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
y
training/Adam/Square_3Square2training/Adam/gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:
*
T0
l
training/Adam/add_10Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:

h
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_9*
T0*
_output_shapes
:

Z
training/Adam/Const_9Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_10Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_10training/Adam/Const_10*
_output_shapes
:
*
T0
Л
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_9*
T0*
_output_shapes
:

`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:

[
training/Adam/add_11/yConst*
dtype0*
_output_shapes
: *
valueB
 *Х┐╓3
n
training/Adam/add_11Addtraining/Adam/Sqrt_4training/Adam/add_11/y*
T0*
_output_shapes
:

s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_11*
_output_shapes
:
*
T0
h
training/Adam/ReadVariableOp_30ReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

z
training/Adam/sub_13Subtraining/Adam/ReadVariableOp_30training/Adam/truediv_4*
T0*
_output_shapes
:

p
 training/Adam/AssignVariableOp_9AssignVariableOptraining/Adam/Variable_3training/Adam/add_9*
dtype0
Ч
training/Adam/ReadVariableOp_31ReadVariableOptraining/Adam/Variable_3!^training/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:

r
!training/Adam/AssignVariableOp_10AssignVariableOptraining/Adam/Variable_7training/Adam/add_10*
dtype0
Ш
training/Adam/ReadVariableOp_32ReadVariableOptraining/Adam/Variable_7"^training/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:

f
!training/Adam/AssignVariableOp_11AssignVariableOpdense_1/biastraining/Adam/sub_13*
dtype0
М
training/Adam/ReadVariableOp_33ReadVariableOpdense_1/bias"^training/Adam/AssignVariableOp_11*
_output_shapes
:
*
dtype0
╓
training_1/group_depsNoOp	^loss/mul^metrics/acc/div_no_nan ^training/Adam/ReadVariableOp_15 ^training/Adam/ReadVariableOp_16 ^training/Adam/ReadVariableOp_17 ^training/Adam/ReadVariableOp_23 ^training/Adam/ReadVariableOp_24 ^training/Adam/ReadVariableOp_25 ^training/Adam/ReadVariableOp_31 ^training/Adam/ReadVariableOp_32 ^training/Adam/ReadVariableOp_33^training/Adam/ReadVariableOp_7^training/Adam/ReadVariableOp_8^training/Adam/ReadVariableOp_9
Z
VarIsInitializedOpVarIsInitializedOptraining/Adam/Variable_1*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
Adam/decay*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense/kernel*
_output_shapes
: 
\
VarIsInitializedOp_3VarIsInitializedOptraining/Adam/Variable_2*
_output_shapes
: 
\
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/Variable_5*
_output_shapes
: 
\
VarIsInitializedOp_5VarIsInitializedOptraining/Adam/Variable_4*
_output_shapes
: 
\
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/Variable_6*
_output_shapes
: 
R
VarIsInitializedOp_7VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
]
VarIsInitializedOp_8VarIsInitializedOptraining/Adam/Variable_10*
_output_shapes
: 
S
VarIsInitializedOp_9VarIsInitializedOpAdam/iterations*
_output_shapes
: 
Q
VarIsInitializedOp_10VarIsInitializedOpdense_1/bias*
_output_shapes
: 
J
VarIsInitializedOp_11VarIsInitializedOpcount*
_output_shapes
: 
J
VarIsInitializedOp_12VarIsInitializedOptotal*
_output_shapes
: 
^
VarIsInitializedOp_13VarIsInitializedOptraining/Adam/Variable_11*
_output_shapes
: 
P
VarIsInitializedOp_14VarIsInitializedOpAdam/beta_1*
_output_shapes
: 
]
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/Variable_3*
_output_shapes
: 
]
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/Variable_8*
_output_shapes
: 
O
VarIsInitializedOp_17VarIsInitializedOp
dense/bias*
_output_shapes
: 
L
VarIsInitializedOp_18VarIsInitializedOpAdam/lr*
_output_shapes
: 
P
VarIsInitializedOp_19VarIsInitializedOpAdam/beta_2*
_output_shapes
: 
[
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/Variable*
_output_shapes
: 
]
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/Variable_9*
_output_shapes
: 
]
VarIsInitializedOp_22VarIsInitializedOptraining/Adam/Variable_7*
_output_shapes
: 
В
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^count/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^total/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
E
AssignVariableOpAssignVariableOptotalPlaceholder*
dtype0
_
ReadVariableOpReadVariableOptotal^AssignVariableOp*
dtype0*
_output_shapes
: 
N
Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 
I
AssignVariableOp_1AssignVariableOpcountPlaceholder_1*
dtype0
c
ReadVariableOp_1ReadVariableOpcount^AssignVariableOp_1*
dtype0*
_output_shapes
: 
A
evaluation/group_depsNoOp	^loss/mul^metrics/acc/div_no_nan
Н
(SGD/iterations/Initializer/initial_valueConst*
value	B	 R *!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
д
SGD/iterationsVarHandleOp*
dtype0	*
	container *
shape: *!
_class
loc:@SGD/iterations*
_output_shapes
: *
shared_nameSGD/iterations
m
/SGD/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/iterations*
_output_shapes
: 
У
SGD/iterations/AssignAssignVariableOpSGD/iterations(SGD/iterations/Initializer/initial_value*
dtype0	*!
_class
loc:@SGD/iterations
М
"SGD/iterations/Read/ReadVariableOpReadVariableOpSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
А
 SGD/lr/Initializer/initial_valueConst*
valueB
 *
╫#<*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
М
SGD/lrVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@SGD/lr*
_output_shapes
: *
shared_nameSGD/lr
]
'SGD/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/lr*
_output_shapes
: 
s
SGD/lr/AssignAssignVariableOpSGD/lr SGD/lr/Initializer/initial_value*
_class
loc:@SGD/lr*
dtype0
t
SGD/lr/Read/ReadVariableOpReadVariableOpSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
М
&SGD/momentum/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
Ю
SGD/momentumVarHandleOp*
shape: *
	container *
_class
loc:@SGD/momentum*
_output_shapes
: *
shared_nameSGD/momentum*
dtype0
i
-SGD/momentum/IsInitialized/VarIsInitializedOpVarIsInitializedOpSGD/momentum*
_output_shapes
: 
Л
SGD/momentum/AssignAssignVariableOpSGD/momentum&SGD/momentum/Initializer/initial_value*
_class
loc:@SGD/momentum*
dtype0
Ж
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
Ж
#SGD/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
Х
	SGD/decayVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@SGD/decay*
_output_shapes
: *
shared_name	SGD/decay
c
*SGD/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOp	SGD/decay*
_output_shapes
: 

SGD/decay/AssignAssignVariableOp	SGD/decay#SGD/decay/Initializer/initial_value*
_class
loc:@SGD/decay*
dtype0
}
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
_class
loc:@SGD/decay*
dtype0
t
	input_1_1Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
P
Shape_1Shape	input_1_1*
T0*
out_type0*
_output_shapes
:
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Г
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
ellipsis_mask *
end_mask *
Index0*
new_axis_mask *
T0
\
Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
         
u
Reshape_1/shapePackstrided_slice_1Reshape_1/shape/1*

axis *
_output_shapes
:*
N*
T0
q
	Reshape_1Reshape	input_1_1Reshape_1/shape*
T0*
Tshape0*(
_output_shapes
:         Р
г
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
:*
valueB"     
Х
-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *м\▒╜*!
_class
loc:@dense_2/kernel
Х
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *м\▒=
ь
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Р*
dtype0*
seed2 *

seed *
T0
╓
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
T0
щ
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Р
█
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Р*
T0
н
dense_2/kernelVarHandleOp*
dtype0*
shape:	Р*
	container *!
_class
loc:@dense_2/kernel*
_output_shapes
: *
shared_namedense_2/kernel
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
Ф
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*!
_class
loc:@dense_2/kernel*
dtype0
Х
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes
:	Р*!
_class
loc:@dense_2/kernel
М
dense_2/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:
в
dense_2/biasVarHandleOp*
dtype0*
shape:*
	container *
_class
loc:@dense_2/bias*
_output_shapes
: *
shared_namedense_2/bias
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
Г
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0*
_class
loc:@dense_2/bias
К
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
_class
loc:@dense_2/bias*
dtype0
g
MatMul_2/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes
:	Р
О
MatMul_2MatMul	Reshape_1MatMul_2/ReadVariableOp*
transpose_b( *'
_output_shapes
:         *
T0*
transpose_a( 
a
BiasAdd_2/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_2BiasAddMatMul_2BiasAdd_2/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         
K
Relu_1Relu	BiasAdd_2*'
_output_shapes
:         *
T0
f
cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
M
cond_1/switch_tIdentitycond_1/Switch:1*
T0
*
_output_shapes
: 
K
cond_1/switch_fIdentitycond_1/Switch*
_output_shapes
: *
T0

Q
cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

j
cond_1/dropout/rateConst^cond_1/switch_t*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: 
q
cond_1/dropout/ShapeShapecond_1/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Э
cond_1/dropout/Shape/SwitchSwitchRelu_1cond_1/pred_id*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         *
T0
k
cond_1/dropout/sub/xConst^cond_1/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
cond_1/dropout/subSubcond_1/dropout/sub/xcond_1/dropout/rate*
_output_shapes
: *
T0
x
!cond_1/dropout/random_uniform/minConst^cond_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_1/dropout/random_uniform/maxConst^cond_1/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
к
+cond_1/dropout/random_uniform/RandomUniformRandomUniformcond_1/dropout/Shape*
dtype0*'
_output_shapes
:         *
seed2 *
T0*

seed 
П
!cond_1/dropout/random_uniform/subSub!cond_1/dropout/random_uniform/max!cond_1/dropout/random_uniform/min*
T0*
_output_shapes
: 
к
!cond_1/dropout/random_uniform/mulMul+cond_1/dropout/random_uniform/RandomUniform!cond_1/dropout/random_uniform/sub*'
_output_shapes
:         *
T0
Ь
cond_1/dropout/random_uniformAdd!cond_1/dropout/random_uniform/mul!cond_1/dropout/random_uniform/min*'
_output_shapes
:         *
T0
~
cond_1/dropout/addAddcond_1/dropout/subcond_1/dropout/random_uniform*
T0*'
_output_shapes
:         
c
cond_1/dropout/FloorFloorcond_1/dropout/add*'
_output_shapes
:         *
T0
Ж
cond_1/dropout/truedivRealDivcond_1/dropout/Shape/Switch:1cond_1/dropout/sub*'
_output_shapes
:         *
T0
y
cond_1/dropout/mulMulcond_1/dropout/truedivcond_1/dropout/Floor*
T0*'
_output_shapes
:         
e
cond_1/IdentityIdentitycond_1/Identity/Switch*'
_output_shapes
:         *
T0
Ш
cond_1/Identity/SwitchSwitchRelu_1cond_1/pred_id*
T0*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         
w
cond_1/MergeMergecond_1/Identitycond_1/dropout/mul*
N*
T0*)
_output_shapes
:         : 
г
/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"   
   *!
_class
loc:@dense_3/kernel*
dtype0
Х
-dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї╛*!
_class
loc:@dense_3/kernel
Х
-dense_3/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї>
ы
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
T0*

seed *!
_class
loc:@dense_3/kernel*
_output_shapes

:
*
dtype0*
seed2 
╓
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
ш
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes

:
*
T0*!
_class
loc:@dense_3/kernel
┌
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:

м
dense_3/kernelVarHandleOp*
dtype0*
shape
:
*
	container *!
_class
loc:@dense_3/kernel*
shared_namedense_3/kernel*
_output_shapes
: 
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Ф
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*!
_class
loc:@dense_3/kernel*
dtype0
Ф
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes

:

М
dense_3/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:

в
dense_3/biasVarHandleOp*
_output_shapes
: *
shared_namedense_3/bias*
dtype0*
shape:
*
	container *
_class
loc:@dense_3/bias
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
Г
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
_class
loc:@dense_3/bias*
dtype0
К
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:

f
MatMul_3/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:
*
dtype0
С
MatMul_3MatMulcond_1/MergeMatMul_3/ReadVariableOp*
transpose_b( *'
_output_shapes
:         
*
T0*
transpose_a( 
a
BiasAdd_3/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_3BiasAddMatMul_3BiasAdd_3/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         

Q
	Softmax_1Softmax	BiasAdd_3*
T0*'
_output_shapes
:         

Ж
output_1_target_1Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
T
Const_1Const*
dtype0*
_output_shapes
:*
valueB*  А?
И
output_1_sample_weights_1PlaceholderWithDefaultConst_1*
dtype0*#
_output_shapes
:         *
shape:         
z
total_1/Initializer/zerosConst*
valueB
 *    *
_class
loc:@total_1*
dtype0*
_output_shapes
: 
П
total_1VarHandleOp*
shape: *
	container *
_class
loc:@total_1*
shared_name	total_1*
_output_shapes
: *
dtype0
_
(total_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_1*
_output_shapes
: 
o
total_1/AssignAssignVariableOptotal_1total_1/Initializer/zeros*
_class
loc:@total_1*
dtype0
w
total_1/Read/ReadVariableOpReadVariableOptotal_1*
dtype0*
_output_shapes
: *
_class
loc:@total_1
z
count_1/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@count_1*
dtype0
П
count_1VarHandleOp*
shape: *
	container *
_class
loc:@count_1*
shared_name	count_1*
_output_shapes
: *
dtype0
_
(count_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_1*
_output_shapes
: 
o
count_1/AssignAssignVariableOpcount_1count_1/Initializer/zeros*
dtype0*
_class
loc:@count_1
w
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_class
loc:@count_1*
dtype0*
_output_shapes
: 
u
"loss_1/output_1_loss/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ъ
loss_1/output_1_loss/ReshapeReshapeoutput_1_target_1"loss_1/output_1_loss/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
М
loss_1/output_1_loss/CastCastloss_1/output_1_loss/Reshape*

SrcT0*
Truncate( *

DstT0	*#
_output_shapes
:         
u
$loss_1/output_1_loss/Reshape_1/shapeConst*
valueB"    
   *
dtype0*
_output_shapes
:
Ъ
loss_1/output_1_loss/Reshape_1Reshape	BiasAdd_3$loss_1/output_1_loss/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:         

Ч
>loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_1/output_1_loss/Cast*
T0	*
out_type0*
_output_shapes
:
О
\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_1/output_1_loss/Reshape_1loss_1/output_1_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":         :         

в
Iloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights_1*
_output_shapes
:*
T0*
out_type0
К
Hloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B :
ф
Hloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Й
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Й
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
В
Eloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualGloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: *
T0
М
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

╒
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentitySloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
╙
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╞
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

ї
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*X
_classN
LJloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

С
qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
Ю
xloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rankRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*Z
_classP
NLloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rank*
_output_shapes
: : 
в
zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank
■
kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

Й
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

З
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitykloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
М
lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
└
Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
valueB :
         *
dtype0
┘
Аloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЛloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
_output_shapes

:*
T0*

Tdim0
╕
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
Ф
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЗloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchlloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::
╟
Еloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╕
Еloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
value	B :*
dtype0
╥
loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЕloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЕloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
┤
Бloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B :
╬
|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Аloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeБloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
N*
T0*

Tidx0*
_output_shapes

:
┬
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
▀
Вloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsНloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

:*
T0*

Tdim0
╝
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeRloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Щ
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЙloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchlloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
е
Оloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationВloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
set_operationa-b*<
_output_shapes*
(:         :         :*
validate_indices(
╙
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeРloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
й
wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstn^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
б
uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЖloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
В
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchqloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranklloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Д
_classz
xvloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Е
jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergemloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
N*
T0
*
_output_shapes
: : 
╚
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergejloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
N*
T0
*
_output_shapes
: : 
й
Aloss_1/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
Т
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Я
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*
_output_shapes
: *,
value#B! Boutput_1_sample_weights_1:0*
dtype0
С
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
т
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*o
valuefBd B^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
О
Closs_1/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
Я
Nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergePloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

╧
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

═
Ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
╬
Oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: *
T0

з
Lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Н
Zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tM^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
Р
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.*
dtype0
ў
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Д
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*,
value#B! Boutput_1_sample_weights_1:0*
dtype0*
_output_shapes
: 
Ў
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *
valueB Bvalues.shape=
╟
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *o
valuefBd B^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0
є
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstQ^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
щ
Nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchUloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
К
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge
Ж
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchIloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*\
_classR
PNloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::
Д
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchHloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*[
_classQ
OMloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
Ў
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchEloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarOloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*X
_classN
LJloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

С
\loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityPloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fO^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*c
_classY
WUloc:@loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╝
Mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMerge\loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
T0
*
_output_shapes
: : 
в
6loss_1/output_1_loss/broadcast_weights/ones_like/ShapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsN^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
:*
T0*
out_type0
╦
6loss_1/output_1_loss/broadcast_weights/ones_like/ConstConstN^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ш
0loss_1/output_1_loss/broadcast_weights/ones_likeFill6loss_1/output_1_loss/broadcast_weights/ones_like/Shape6loss_1/output_1_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0*

index_type0
и
&loss_1/output_1_loss/broadcast_weightsMuloutput_1_sample_weights_10loss_1/output_1_loss/broadcast_weights/ones_like*#
_output_shapes
:         *
T0
╙
loss_1/output_1_loss/MulMul\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&loss_1/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0
d
loss_1/output_1_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
У
loss_1/output_1_loss/SumSumloss_1/output_1_loss/Mulloss_1/output_1_loss/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
f
loss_1/output_1_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
е
loss_1/output_1_loss/Sum_1Sum&loss_1/output_1_loss/broadcast_weightsloss_1/output_1_loss/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
В
loss_1/output_1_loss/div_no_nanDivNoNanloss_1/output_1_loss/Sumloss_1/output_1_loss/Sum_1*
T0*
_output_shapes
: 
_
loss_1/output_1_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
Ю
loss_1/output_1_loss/MeanMeanloss_1/output_1_loss/div_no_nanloss_1/output_1_loss/Const_2*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
Q
loss_1/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
[

loss_1/mulMulloss_1/mul/xloss_1/output_1_loss/Mean*
T0*
_output_shapes
: 
З
metrics_1/acc/CastCastoutput_1_target_1*

SrcT0*
Truncate( *

DstT0*0
_output_shapes
:                  
В
metrics_1/acc/SqueezeSqueezemetrics_1/acc/Cast*
squeeze_dims

         *#
_output_shapes
:         *
T0
i
metrics_1/acc/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
         
Ц
metrics_1/acc/ArgMaxArgMax	Softmax_1metrics_1/acc/ArgMax/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:         

metrics_1/acc/Cast_1Castmetrics_1/acc/ArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:         
w
metrics_1/acc/EqualEqualmetrics_1/acc/Squeezemetrics_1/acc/Cast_1*#
_output_shapes
:         *
T0
~
metrics_1/acc/Cast_2Castmetrics_1/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:         
a
metrics_1/acc/SizeSizemetrics_1/acc/Cast_2*
T0*
out_type0*
_output_shapes
: 
p
metrics_1/acc/Cast_3Castmetrics_1/acc/Size*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 
]
metrics_1/acc/ConstConst*
_output_shapes
:*
valueB: *
dtype0
Б
metrics_1/acc/SumSummetrics_1/acc/Cast_2metrics_1/acc/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
a
!metrics_1/acc/AssignAddVariableOpAssignAddVariableOptotal_1metrics_1/acc/Sum*
dtype0
А
metrics_1/acc/ReadVariableOpReadVariableOptotal_1"^metrics_1/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
Е
#metrics_1/acc/AssignAddVariableOp_1AssignAddVariableOpcount_1metrics_1/acc/Cast_3^metrics_1/acc/ReadVariableOp*
dtype0
г
metrics_1/acc/ReadVariableOp_1ReadVariableOpcount_1$^metrics_1/acc/AssignAddVariableOp_1^metrics_1/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
И
'metrics_1/acc/div_no_nan/ReadVariableOpReadVariableOptotal_1^metrics_1/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
К
)metrics_1/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_1^metrics_1/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Щ
metrics_1/acc/div_no_nanDivNoNan'metrics_1/acc/div_no_nan/ReadVariableOp)metrics_1/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Г
metrics_1/acc/Squeeze_1Squeezeoutput_1_target_1*#
_output_shapes
:         *
T0*
squeeze_dims

         
k
 metrics_1/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
metrics_1/acc/ArgMax_1ArgMax	Softmax_1 metrics_1/acc/ArgMax_1/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:         
Б
metrics_1/acc/Cast_4Castmetrics_1/acc/ArgMax_1*

DstT0*#
_output_shapes
:         *

SrcT0	*
Truncate( 
{
metrics_1/acc/Equal_1Equalmetrics_1/acc/Squeeze_1metrics_1/acc/Cast_4*#
_output_shapes
:         *
T0
А
metrics_1/acc/Cast_5Castmetrics_1/acc/Equal_1*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:         
_
metrics_1/acc/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Е
metrics_1/acc/MeanMeanmetrics_1/acc/Cast_5metrics_1/acc/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
А
training_2/SGD/gradients/ShapeConst*
valueB *
_class
loc:@loss_1/mul*
dtype0*
_output_shapes
: 
Ж
"training_2/SGD/gradients/grad_ys_0Const*
valueB
 *  А?*
_class
loc:@loss_1/mul*
dtype0*
_output_shapes
: 
╗
training_2/SGD/gradients/FillFilltraining_2/SGD/gradients/Shape"training_2/SGD/gradients/grad_ys_0*

index_type0*
_class
loc:@loss_1/mul*
_output_shapes
: *
T0
н
,training_2/SGD/gradients/loss_1/mul_grad/MulMultraining_2/SGD/gradients/Fillloss_1/output_1_loss/Mean*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
в
.training_2/SGD/gradients/loss_1/mul_grad/Mul_1Multraining_2/SGD/gradients/Fillloss_1/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss_1/mul
╢
Etraining_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB *,
_class"
 loc:@loss_1/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ю
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/ReshapeReshape.training_2/SGD/gradients/loss_1/mul_grad/Mul_1Etraining_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape/shape*,
_class"
 loc:@loss_1/output_1_loss/Mean*
Tshape0*
_output_shapes
: *
T0
о
=training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/ConstConst*
_output_shapes
: *
valueB *,
_class"
 loc:@loss_1/output_1_loss/Mean*
dtype0
е
<training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/TileTile?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Reshape=training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const*
T0*

Tmultiples0*,
_class"
 loc:@loss_1/output_1_loss/Mean*
_output_shapes
: 
▓
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const_1Const*
valueB
 *  А?*,
_class"
 loc:@loss_1/output_1_loss/Mean*
dtype0*
_output_shapes
: 
Ш
?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivRealDiv<training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Tile?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/Const_1*
T0*,
_class"
 loc:@loss_1/output_1_loss/Mean*
_output_shapes
: 
║
Ctraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ShapeConst*
valueB *2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
╝
Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1Const*
valueB *2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
щ
Straining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ShapeEtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*2
_output_shapes 
:         :         
Ж
Htraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivloss_1/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: 
┘
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/SumSumHtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nanStraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
	keep_dims( *

Tidx0*
_output_shapes
: 
╗
Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ReshapeReshapeAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/SumCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: 
╖
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/NegNegloss_1/output_1_loss/Sum*
_output_shapes
: *
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan
К
Jtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNanAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Negloss_1/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: 
У
Jtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanJtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_1loss_1/output_1_loss/Sum_1*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
к
Atraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/mulMul?training_2/SGD/gradients/loss_1/output_1_loss/Mean_grad/truedivJtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/div_no_nan_2*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
_output_shapes
: *
T0
╓
Ctraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Sum_1SumAtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/mulUtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
┴
Gtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Reshape_1ReshapeCtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Sum_1Etraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/Shape_1*
T0*2
_class(
&$loc:@loss_1/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: 
╗
Dtraining_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss_1/output_1_loss/Sum*
dtype0*
_output_shapes
:
╢
>training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/ReshapeReshapeEtraining_2/SGD/gradients/loss_1/output_1_loss/div_no_nan_grad/ReshapeDtraining_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss_1/output_1_loss/Sum*
Tshape0
┴
<training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/ShapeShapeloss_1/output_1_loss/Mul*+
_class!
loc:@loss_1/output_1_loss/Sum*
_output_shapes
:*
T0*
out_type0
о
;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/TileTile>training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Reshape<training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Shape*

Tmultiples0*+
_class!
loc:@loss_1/output_1_loss/Sum*#
_output_shapes
:         *
T0
Е
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ShapeShape\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*+
_class!
loc:@loss_1/output_1_loss/Mul*
_output_shapes
:
╤
>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1Shape&loss_1/output_1_loss/broadcast_weights*
T0*
out_type0*+
_class!
loc:@loss_1/output_1_loss/Mul*
_output_shapes
:
═
Ltraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1*+
_class!
loc:@loss_1/output_1_loss/Mul*2
_output_shapes 
:         :         *
T0
Б
:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/MulMul;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Tile&loss_1/output_1_loss/broadcast_weights*
T0*+
_class!
loc:@loss_1/output_1_loss/Mul*#
_output_shapes
:         
╕
:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/SumSum:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/MulLtraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0*+
_class!
loc:@loss_1/output_1_loss/Mul
м
>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ReshapeReshape:training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape*#
_output_shapes
:         *
T0*+
_class!
loc:@loss_1/output_1_loss/Mul*
Tshape0
╣
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Mul_1Mul\loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits;training_2/SGD/gradients/loss_1/output_1_loss/Sum_grad/Tile*
T0*+
_class!
loc:@loss_1/output_1_loss/Mul*#
_output_shapes
:         
╛
<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum_1Sum<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Mul_1Ntraining_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/BroadcastGradientArgs:1*
T0*+
_class!
loc:@loss_1/output_1_loss/Mul*
	keep_dims( *

Tidx0*
_output_shapes
:
▓
@training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Reshape_1Reshape<training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Sum_1>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/Shape_1*+
_class!
loc:@loss_1/output_1_loss/Mul*
Tshape0*#
_output_shapes
:         *
T0
│
#training_2/SGD/gradients/zeros_like	ZerosLike^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

╪
Кtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient^loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:         
*
T0*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╞
Йtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0*
_output_shapes
: 
П
Еtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims>training_2/SGD/gradients/loss_1/output_1_loss/Mul_grad/ReshapeЙtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:         *
T0*

Tdim0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
╜
~training_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulЕtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsКtraining_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*o
_classe
caloc:@loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

╛
Btraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_3*
T0*
out_type0*1
_class'
%#loc:@loss_1/output_1_loss/Reshape_1*
_output_shapes
:
Ж
Dtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ReshapeReshape~training_2/SGD/gradients/loss_1/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulBtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Shape*1
_class'
%#loc:@loss_1/output_1_loss/Reshape_1*
Tshape0*'
_output_shapes
:         
*
T0
т
3training_2/SGD/gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Reshape*
data_formatNHWC*
_class
loc:@BiasAdd_3*
_output_shapes
:
*
T0
Л
-training_2/SGD/gradients/MatMul_3_grad/MatMulMatMulDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/ReshapeMatMul_3/ReadVariableOp*
transpose_a( *
T0*
_class
loc:@MatMul_3*
transpose_b(*'
_output_shapes
:         
∙
/training_2/SGD/gradients/MatMul_3_grad/MatMul_1MatMulcond_1/MergeDtraining_2/SGD/gradients/loss_1/output_1_loss/Reshape_1_grad/Reshape*
transpose_a(*
T0*
_class
loc:@MatMul_3*
transpose_b( *
_output_shapes

:

▀
4training_2/SGD/gradients/cond_1/Merge_grad/cond_gradSwitch-training_2/SGD/gradients/MatMul_3_grad/MatMulcond_1/pred_id*
T0*
_class
loc:@MatMul_3*:
_output_shapes(
&:         :         
│
6training_2/SGD/gradients/cond_1/dropout/mul_grad/ShapeShapecond_1/dropout/truediv*
_output_shapes
:*
T0*
out_type0*%
_class
loc:@cond_1/dropout/mul
│
8training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1Shapecond_1/dropout/Floor*
T0*
out_type0*%
_class
loc:@cond_1/dropout/mul*
_output_shapes
:
╡
Ftraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape8training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0*%
_class
loc:@cond_1/dropout/mul
т
4training_2/SGD/gradients/cond_1/dropout/mul_grad/MulMul6training_2/SGD/gradients/cond_1/Merge_grad/cond_grad:1cond_1/dropout/Floor*'
_output_shapes
:         *
T0*%
_class
loc:@cond_1/dropout/mul
а
4training_2/SGD/gradients/cond_1/dropout/mul_grad/SumSum4training_2/SGD/gradients/cond_1/dropout/mul_grad/MulFtraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_1/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:
Ш
8training_2/SGD/gradients/cond_1/dropout/mul_grad/ReshapeReshape4training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum6training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape*
T0*%
_class
loc:@cond_1/dropout/mul*
Tshape0*'
_output_shapes
:         
ц
6training_2/SGD/gradients/cond_1/dropout/mul_grad/Mul_1Mulcond_1/dropout/truediv6training_2/SGD/gradients/cond_1/Merge_grad/cond_grad:1*'
_output_shapes
:         *
T0*%
_class
loc:@cond_1/dropout/mul
ж
6training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum_1Sum6training_2/SGD/gradients/cond_1/dropout/mul_grad/Mul_1Htraining_2/SGD/gradients/cond_1/dropout/mul_grad/BroadcastGradientArgs:1*%
_class
loc:@cond_1/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Ю
:training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshape_1Reshape6training_2/SGD/gradients/cond_1/dropout/mul_grad/Sum_18training_2/SGD/gradients/cond_1/dropout/mul_grad/Shape_1*%
_class
loc:@cond_1/dropout/mul*
Tshape0*'
_output_shapes
:         *
T0
б
training_2/SGD/gradients/SwitchSwitchRelu_1cond_1/pred_id*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         *
T0
Э
!training_2/SGD/gradients/IdentityIdentity!training_2/SGD/gradients/Switch:1*
_class
loc:@Relu_1*'
_output_shapes
:         *
T0
Ь
 training_2/SGD/gradients/Shape_1Shape!training_2/SGD/gradients/Switch:1*
_output_shapes
:*
T0*
out_type0*
_class
loc:@Relu_1
и
$training_2/SGD/gradients/zeros/ConstConst"^training_2/SGD/gradients/Identity*
_class
loc:@Relu_1*
dtype0*
_output_shapes
: *
valueB
 *    
═
training_2/SGD/gradients/zerosFill training_2/SGD/gradients/Shape_1$training_2/SGD/gradients/zeros/Const*
T0*

index_type0*
_class
loc:@Relu_1*'
_output_shapes
:         
ї
>training_2/SGD/gradients/cond_1/Identity/Switch_grad/cond_gradMerge4training_2/SGD/gradients/cond_1/Merge_grad/cond_gradtraining_2/SGD/gradients/zeros*)
_output_shapes
:         : *
T0*
N*
_class
loc:@Relu_1
┬
:training_2/SGD/gradients/cond_1/dropout/truediv_grad/ShapeShapecond_1/dropout/Shape/Switch:1*)
_class
loc:@cond_1/dropout/truediv*
_output_shapes
:*
T0*
out_type0
к
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *)
_class
loc:@cond_1/dropout/truediv*
dtype0
┼
Jtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1*)
_class
loc:@cond_1/dropout/truediv*2
_output_shapes 
:         :         *
T0
Є
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDivRealDiv8training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshapecond_1/dropout/sub*
T0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
┤
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/SumSum<training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDivJtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0*)
_class
loc:@cond_1/dropout/truediv
и
<training_2/SGD/gradients/cond_1/dropout/truediv_grad/ReshapeReshape8training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape*
T0*)
_class
loc:@cond_1/dropout/truediv*
Tshape0*'
_output_shapes
:         
╗
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/NegNegcond_1/dropout/Shape/Switch:1*'
_output_shapes
:         *
T0*)
_class
loc:@cond_1/dropout/truediv
Ї
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_1RealDiv8training_2/SGD/gradients/cond_1/dropout/truediv_grad/Negcond_1/dropout/sub*
T0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
·
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_2RealDiv>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_1cond_1/dropout/sub*
T0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
Ц
8training_2/SGD/gradients/cond_1/dropout/truediv_grad/mulMul8training_2/SGD/gradients/cond_1/dropout/mul_grad/Reshape>training_2/SGD/gradients/cond_1/dropout/truediv_grad/RealDiv_2*
T0*)
_class
loc:@cond_1/dropout/truediv*'
_output_shapes
:         
┤
:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum_1Sum8training_2/SGD/gradients/cond_1/dropout/truediv_grad/mulLtraining_2/SGD/gradients/cond_1/dropout/truediv_grad/BroadcastGradientArgs:1*)
_class
loc:@cond_1/dropout/truediv*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
Э
>training_2/SGD/gradients/cond_1/dropout/truediv_grad/Reshape_1Reshape:training_2/SGD/gradients/cond_1/dropout/truediv_grad/Sum_1<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*)
_class
loc:@cond_1/dropout/truediv*
Tshape0
г
!training_2/SGD/gradients/Switch_1SwitchRelu_1cond_1/pred_id*
_class
loc:@Relu_1*:
_output_shapes(
&:         :         *
T0
Я
#training_2/SGD/gradients/Identity_1Identity!training_2/SGD/gradients/Switch_1*
_class
loc:@Relu_1*'
_output_shapes
:         *
T0
Ь
 training_2/SGD/gradients/Shape_2Shape!training_2/SGD/gradients/Switch_1*
T0*
out_type0*
_class
loc:@Relu_1*
_output_shapes
:
м
&training_2/SGD/gradients/zeros_1/ConstConst$^training_2/SGD/gradients/Identity_1*
valueB
 *    *
_class
loc:@Relu_1*
dtype0*
_output_shapes
: 
╤
 training_2/SGD/gradients/zeros_1Fill training_2/SGD/gradients/Shape_2&training_2/SGD/gradients/zeros_1/Const*'
_output_shapes
:         *
T0*

index_type0*
_class
loc:@Relu_1
Д
Ctraining_2/SGD/gradients/cond_1/dropout/Shape/Switch_grad/cond_gradMerge training_2/SGD/gradients/zeros_1<training_2/SGD/gradients/cond_1/dropout/truediv_grad/Reshape*
T0*
N*
_class
loc:@Relu_1*)
_output_shapes
:         : 
А
training_2/SGD/gradients/AddNAddN>training_2/SGD/gradients/cond_1/Identity/Switch_grad/cond_gradCtraining_2/SGD/gradients/cond_1/dropout/Shape/Switch_grad/cond_grad*
T0*
N*
_class
loc:@Relu_1*'
_output_shapes
:         
н
-training_2/SGD/gradients/Relu_1_grad/ReluGradReluGradtraining_2/SGD/gradients/AddNRelu_1*
T0*
_class
loc:@Relu_1*'
_output_shapes
:         
╦
3training_2/SGD/gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad-training_2/SGD/gradients/Relu_1_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC*
_class
loc:@BiasAdd_2
ї
-training_2/SGD/gradients/MatMul_2_grad/MatMulMatMul-training_2/SGD/gradients/Relu_1_grad/ReluGradMatMul_2/ReadVariableOp*
transpose_b(*(
_output_shapes
:         Р*
transpose_a( *
T0*
_class
loc:@MatMul_2
р
/training_2/SGD/gradients/MatMul_2_grad/MatMul_1MatMul	Reshape_1-training_2/SGD/gradients/Relu_1_grad/ReluGrad*
_class
loc:@MatMul_2*
transpose_b( *
_output_shapes
:	Р*
transpose_a(*
T0
V
training_2/SGD/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
l
"training_2/SGD/AssignAddVariableOpAssignAddVariableOpSGD/iterationstraining_2/SGD/Const*
dtype0	
Й
training_2/SGD/ReadVariableOpReadVariableOpSGD/iterations#^training_2/SGD/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
u
$training_2/SGD/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"     *
dtype0
_
training_2/SGD/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ъ
training_2/SGD/zerosFill$training_2/SGD/zeros/shape_as_tensortraining_2/SGD/zeros/Const*
_output_shapes
:	Р*
T0*

index_type0
╚
training_2/SGD/VariableVarHandleOp*
dtype0*
shape:	Р*
	container **
_class 
loc:@training_2/SGD/Variable*
_output_shapes
: *(
shared_nametraining_2/SGD/Variable

8training_2/SGD/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable*
_output_shapes
: 
Ъ
training_2/SGD/Variable/AssignAssignVariableOptraining_2/SGD/Variabletraining_2/SGD/zeros**
_class 
loc:@training_2/SGD/Variable*
dtype0
░
+training_2/SGD/Variable/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable**
_class 
loc:@training_2/SGD/Variable*
dtype0*
_output_shapes
:	Р
c
training_2/SGD/zeros_1Const*
dtype0*
_output_shapes
:*
valueB*    
╔
training_2/SGD/Variable_1VarHandleOp*
_output_shapes
: **
shared_nametraining_2/SGD/Variable_1*
dtype0*
shape:*
	container *,
_class"
 loc:@training_2/SGD/Variable_1
Г
:training_2/SGD/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_1*
_output_shapes
: 
в
 training_2/SGD/Variable_1/AssignAssignVariableOptraining_2/SGD/Variable_1training_2/SGD/zeros_1*,
_class"
 loc:@training_2/SGD/Variable_1*
dtype0
▒
-training_2/SGD/Variable_1/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_1*
dtype0*
_output_shapes
:*,
_class"
 loc:@training_2/SGD/Variable_1
k
training_2/SGD/zeros_2Const*
dtype0*
_output_shapes

:
*
valueB
*    
═
training_2/SGD/Variable_2VarHandleOp*
dtype0*
shape
:
*
	container *,
_class"
 loc:@training_2/SGD/Variable_2**
shared_nametraining_2/SGD/Variable_2*
_output_shapes
: 
Г
:training_2/SGD/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_2*
_output_shapes
: 
в
 training_2/SGD/Variable_2/AssignAssignVariableOptraining_2/SGD/Variable_2training_2/SGD/zeros_2*,
_class"
 loc:@training_2/SGD/Variable_2*
dtype0
╡
-training_2/SGD/Variable_2/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_2*,
_class"
 loc:@training_2/SGD/Variable_2*
dtype0*
_output_shapes

:

c
training_2/SGD/zeros_3Const*
dtype0*
_output_shapes
:
*
valueB
*    
╔
training_2/SGD/Variable_3VarHandleOp*
dtype0*
shape:
*
	container *,
_class"
 loc:@training_2/SGD/Variable_3**
shared_nametraining_2/SGD/Variable_3*
_output_shapes
: 
Г
:training_2/SGD/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_2/SGD/Variable_3*
_output_shapes
: 
в
 training_2/SGD/Variable_3/AssignAssignVariableOptraining_2/SGD/Variable_3training_2/SGD/zeros_3*,
_class"
 loc:@training_2/SGD/Variable_3*
dtype0
▒
-training_2/SGD/Variable_3/Read/ReadVariableOpReadVariableOptraining_2/SGD/Variable_3*
dtype0*
_output_shapes
:
*,
_class"
 loc:@training_2/SGD/Variable_3
d
training_2/SGD/ReadVariableOp_1ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
z
!training_2/SGD/mul/ReadVariableOpReadVariableOptraining_2/SGD/Variable*
dtype0*
_output_shapes
:	Р
З
training_2/SGD/mulMultraining_2/SGD/ReadVariableOp_1!training_2/SGD/mul/ReadVariableOp*
_output_shapes
:	Р*
T0
^
training_2/SGD/ReadVariableOp_2ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ч
training_2/SGD/mul_1Multraining_2/SGD/ReadVariableOp_2/training_2/SGD/gradients/MatMul_2_grad/MatMul_1*
T0*
_output_shapes
:	Р
m
training_2/SGD/subSubtraining_2/SGD/multraining_2/SGD/mul_1*
_output_shapes
:	Р*
T0
m
training_2/SGD/AssignVariableOpAssignVariableOptraining_2/SGD/Variabletraining_2/SGD/sub*
dtype0
Ъ
training_2/SGD/ReadVariableOp_3ReadVariableOptraining_2/SGD/Variable ^training_2/SGD/AssignVariableOp*
dtype0*
_output_shapes
:	Р
o
training_2/SGD/ReadVariableOp_4ReadVariableOpdense_2/kernel*
_output_shapes
:	Р*
dtype0
x
training_2/SGD/addAddtraining_2/SGD/ReadVariableOp_4training_2/SGD/sub*
_output_shapes
:	Р*
T0
f
!training_2/SGD/AssignVariableOp_1AssignVariableOpdense_2/kerneltraining_2/SGD/add*
dtype0
У
training_2/SGD/ReadVariableOp_5ReadVariableOpdense_2/kernel"^training_2/SGD/AssignVariableOp_1*
dtype0*
_output_shapes
:	Р
d
training_2/SGD/ReadVariableOp_6ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
y
#training_2/SGD/mul_2/ReadVariableOpReadVariableOptraining_2/SGD/Variable_1*
dtype0*
_output_shapes
:
Ж
training_2/SGD/mul_2Multraining_2/SGD/ReadVariableOp_6#training_2/SGD/mul_2/ReadVariableOp*
_output_shapes
:*
T0
^
training_2/SGD/ReadVariableOp_7ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ц
training_2/SGD/mul_3Multraining_2/SGD/ReadVariableOp_73training_2/SGD/gradients/BiasAdd_2_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training_2/SGD/sub_1Subtraining_2/SGD/mul_2training_2/SGD/mul_3*
T0*
_output_shapes
:
s
!training_2/SGD/AssignVariableOp_2AssignVariableOptraining_2/SGD/Variable_1training_2/SGD/sub_1*
dtype0
Щ
training_2/SGD/ReadVariableOp_8ReadVariableOptraining_2/SGD/Variable_1"^training_2/SGD/AssignVariableOp_2*
_output_shapes
:*
dtype0
h
training_2/SGD/ReadVariableOp_9ReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
w
training_2/SGD/add_1Addtraining_2/SGD/ReadVariableOp_9training_2/SGD/sub_1*
T0*
_output_shapes
:
f
!training_2/SGD/AssignVariableOp_3AssignVariableOpdense_2/biastraining_2/SGD/add_1*
dtype0
Н
 training_2/SGD/ReadVariableOp_10ReadVariableOpdense_2/bias"^training_2/SGD/AssignVariableOp_3*
dtype0*
_output_shapes
:
e
 training_2/SGD/ReadVariableOp_11ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
}
#training_2/SGD/mul_4/ReadVariableOpReadVariableOptraining_2/SGD/Variable_2*
dtype0*
_output_shapes

:

Л
training_2/SGD/mul_4Mul training_2/SGD/ReadVariableOp_11#training_2/SGD/mul_4/ReadVariableOp*
T0*
_output_shapes

:

_
 training_2/SGD/ReadVariableOp_12ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ч
training_2/SGD/mul_5Mul training_2/SGD/ReadVariableOp_12/training_2/SGD/gradients/MatMul_3_grad/MatMul_1*
T0*
_output_shapes

:

p
training_2/SGD/sub_2Subtraining_2/SGD/mul_4training_2/SGD/mul_5*
T0*
_output_shapes

:

s
!training_2/SGD/AssignVariableOp_4AssignVariableOptraining_2/SGD/Variable_2training_2/SGD/sub_2*
dtype0
Ю
 training_2/SGD/ReadVariableOp_13ReadVariableOptraining_2/SGD/Variable_2"^training_2/SGD/AssignVariableOp_4*
dtype0*
_output_shapes

:

o
 training_2/SGD/ReadVariableOp_14ReadVariableOpdense_3/kernel*
dtype0*
_output_shapes

:

|
training_2/SGD/add_2Add training_2/SGD/ReadVariableOp_14training_2/SGD/sub_2*
T0*
_output_shapes

:

h
!training_2/SGD/AssignVariableOp_5AssignVariableOpdense_3/kerneltraining_2/SGD/add_2*
dtype0
У
 training_2/SGD/ReadVariableOp_15ReadVariableOpdense_3/kernel"^training_2/SGD/AssignVariableOp_5*
dtype0*
_output_shapes

:

e
 training_2/SGD/ReadVariableOp_16ReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
y
#training_2/SGD/mul_6/ReadVariableOpReadVariableOptraining_2/SGD/Variable_3*
dtype0*
_output_shapes
:

З
training_2/SGD/mul_6Mul training_2/SGD/ReadVariableOp_16#training_2/SGD/mul_6/ReadVariableOp*
_output_shapes
:
*
T0
_
 training_2/SGD/ReadVariableOp_17ReadVariableOpSGD/lr*
dtype0*
_output_shapes
: 
Ч
training_2/SGD/mul_7Mul training_2/SGD/ReadVariableOp_173training_2/SGD/gradients/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:
*
T0
l
training_2/SGD/sub_3Subtraining_2/SGD/mul_6training_2/SGD/mul_7*
T0*
_output_shapes
:

s
!training_2/SGD/AssignVariableOp_6AssignVariableOptraining_2/SGD/Variable_3training_2/SGD/sub_3*
dtype0
Ъ
 training_2/SGD/ReadVariableOp_18ReadVariableOptraining_2/SGD/Variable_3"^training_2/SGD/AssignVariableOp_6*
dtype0*
_output_shapes
:

i
 training_2/SGD/ReadVariableOp_19ReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:

x
training_2/SGD/add_3Add training_2/SGD/ReadVariableOp_19training_2/SGD/sub_3*
T0*
_output_shapes
:

f
!training_2/SGD/AssignVariableOp_7AssignVariableOpdense_3/biastraining_2/SGD/add_3*
dtype0
Н
 training_2/SGD/ReadVariableOp_20ReadVariableOpdense_3/bias"^training_2/SGD/AssignVariableOp_7*
dtype0*
_output_shapes
:

·
training_3/group_depsNoOp^loss_1/mul^metrics_1/acc/div_no_nan^training_2/SGD/ReadVariableOp!^training_2/SGD/ReadVariableOp_10!^training_2/SGD/ReadVariableOp_13!^training_2/SGD/ReadVariableOp_15!^training_2/SGD/ReadVariableOp_18!^training_2/SGD/ReadVariableOp_20 ^training_2/SGD/ReadVariableOp_3 ^training_2/SGD/ReadVariableOp_5 ^training_2/SGD/ReadVariableOp_8
\
VarIsInitializedOp_23VarIsInitializedOptraining_2/SGD/Variable*
_output_shapes
: 
^
VarIsInitializedOp_24VarIsInitializedOptraining_2/SGD/Variable_1*
_output_shapes
: 
^
VarIsInitializedOp_25VarIsInitializedOptraining_2/SGD/Variable_2*
_output_shapes
: 
S
VarIsInitializedOp_26VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_27VarIsInitializedOpdense_2/bias*
_output_shapes
: 
Q
VarIsInitializedOp_28VarIsInitializedOpdense_3/bias*
_output_shapes
: 
L
VarIsInitializedOp_29VarIsInitializedOptotal_1*
_output_shapes
: 
S
VarIsInitializedOp_30VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_31VarIsInitializedOpSGD/momentum*
_output_shapes
: 
K
VarIsInitializedOp_32VarIsInitializedOpSGD/lr*
_output_shapes
: 
S
VarIsInitializedOp_33VarIsInitializedOpSGD/iterations*
_output_shapes
: 
^
VarIsInitializedOp_34VarIsInitializedOptraining_2/SGD/Variable_3*
_output_shapes
: 
L
VarIsInitializedOp_35VarIsInitializedOpcount_1*
_output_shapes
: 
N
VarIsInitializedOp_36VarIsInitializedOp	SGD/decay*
_output_shapes
: 
ч
init_1NoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^count_1/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^total_1/Assign^training_2/SGD/Variable/Assign!^training_2/SGD/Variable_1/Assign!^training_2/SGD/Variable_2/Assign!^training_2/SGD/Variable_3/Assign
N
Placeholder_2Placeholder*
shape: *
dtype0*
_output_shapes
: 
K
AssignVariableOp_2AssignVariableOptotal_1Placeholder_2*
dtype0
e
ReadVariableOp_2ReadVariableOptotal_1^AssignVariableOp_2*
dtype0*
_output_shapes
: 
N
Placeholder_3Placeholder*
shape: *
dtype0*
_output_shapes
: 
K
AssignVariableOp_3AssignVariableOpcount_1Placeholder_3*
dtype0
e
ReadVariableOp_3ReadVariableOpcount_1^AssignVariableOp_3*
dtype0*
_output_shapes
: 
G
evaluation_1/group_depsNoOp^loss_1/mul^metrics_1/acc/div_no_nan
У
+Adam_1/iterations/Initializer/initial_valueConst*
value	B	 R *$
_class
loc:@Adam_1/iterations*
dtype0	*
_output_shapes
: 
н
Adam_1/iterationsVarHandleOp*
dtype0	*
shape: *
	container *$
_class
loc:@Adam_1/iterations*
_output_shapes
: *"
shared_nameAdam_1/iterations
s
2Adam_1/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/iterations*
_output_shapes
: 
Я
Adam_1/iterations/AssignAssignVariableOpAdam_1/iterations+Adam_1/iterations/Initializer/initial_value*$
_class
loc:@Adam_1/iterations*
dtype0	
Х
%Adam_1/iterations/Read/ReadVariableOpReadVariableOpAdam_1/iterations*$
_class
loc:@Adam_1/iterations*
dtype0	*
_output_shapes
: 
Ж
#Adam_1/lr/Initializer/initial_valueConst*
valueB
 *oГ:*
_class
loc:@Adam_1/lr*
dtype0*
_output_shapes
: 
Х
	Adam_1/lrVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam_1/lr*
shared_name	Adam_1/lr*
_output_shapes
: 
c
*Adam_1/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOp	Adam_1/lr*
_output_shapes
: 

Adam_1/lr/AssignAssignVariableOp	Adam_1/lr#Adam_1/lr/Initializer/initial_value*
_class
loc:@Adam_1/lr*
dtype0
}
Adam_1/lr/Read/ReadVariableOpReadVariableOp	Adam_1/lr*
_class
loc:@Adam_1/lr*
dtype0*
_output_shapes
: 
О
'Adam_1/beta_1/Initializer/initial_valueConst*
valueB
 *fff?* 
_class
loc:@Adam_1/beta_1*
dtype0*
_output_shapes
: 
б
Adam_1/beta_1VarHandleOp*
dtype0*
shape: *
	container * 
_class
loc:@Adam_1/beta_1*
_output_shapes
: *
shared_nameAdam_1/beta_1
k
.Adam_1/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/beta_1*
_output_shapes
: 
П
Adam_1/beta_1/AssignAssignVariableOpAdam_1/beta_1'Adam_1/beta_1/Initializer/initial_value* 
_class
loc:@Adam_1/beta_1*
dtype0
Й
!Adam_1/beta_1/Read/ReadVariableOpReadVariableOpAdam_1/beta_1* 
_class
loc:@Adam_1/beta_1*
dtype0*
_output_shapes
: 
О
'Adam_1/beta_2/Initializer/initial_valueConst*
valueB
 *w╛?* 
_class
loc:@Adam_1/beta_2*
dtype0*
_output_shapes
: 
б
Adam_1/beta_2VarHandleOp*
shape: *
	container * 
_class
loc:@Adam_1/beta_2*
shared_nameAdam_1/beta_2*
_output_shapes
: *
dtype0
k
.Adam_1/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/beta_2*
_output_shapes
: 
П
Adam_1/beta_2/AssignAssignVariableOpAdam_1/beta_2'Adam_1/beta_2/Initializer/initial_value* 
_class
loc:@Adam_1/beta_2*
dtype0
Й
!Adam_1/beta_2/Read/ReadVariableOpReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: * 
_class
loc:@Adam_1/beta_2
М
&Adam_1/decay/Initializer/initial_valueConst*
valueB
 *    *
_class
loc:@Adam_1/decay*
dtype0*
_output_shapes
: 
Ю
Adam_1/decayVarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@Adam_1/decay*
_output_shapes
: *
shared_nameAdam_1/decay
i
-Adam_1/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOpAdam_1/decay*
_output_shapes
: 
Л
Adam_1/decay/AssignAssignVariableOpAdam_1/decay&Adam_1/decay/Initializer/initial_value*
_class
loc:@Adam_1/decay*
dtype0
Ж
 Adam_1/decay/Read/ReadVariableOpReadVariableOpAdam_1/decay*
dtype0*
_output_shapes
: *
_class
loc:@Adam_1/decay
t
	input_1_2Placeholder* 
shape:         *
dtype0*+
_output_shapes
:         
P
Shape_2Shape	input_1_2*
_output_shapes
:*
T0*
out_type0
_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
strided_slice_2StridedSliceShape_2strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
Index0*
new_axis_mask *
T0*
_output_shapes
: *

begin_mask *
shrink_axis_mask*
ellipsis_mask *
end_mask 
\
Reshape_2/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
u
Reshape_2/shapePackstrided_slice_2Reshape_2/shape/1*

axis *
_output_shapes
:*
N*
T0
q
	Reshape_2Reshape	input_1_2Reshape_2/shape*(
_output_shapes
:         Р*
T0*
Tshape0
г
/dense_4/kernel/Initializer/random_uniform/shapeConst*
valueB"     *!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
:
Х
-dense_4/kernel/Initializer/random_uniform/minConst*
valueB
 *м\▒╜*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
Х
-dense_4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *м\▒=*!
_class
loc:@dense_4/kernel*
dtype0
ь
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
:	Р
╓
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
_output_shapes
: *
T0
щ
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Р*
T0*!
_class
loc:@dense_4/kernel
█
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes
:	Р
н
dense_4/kernelVarHandleOp*
shape:	Р*
	container *!
_class
loc:@dense_4/kernel*
shared_namedense_4/kernel*
_output_shapes
: *
dtype0
m
/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Ф
dense_4/kernel/AssignAssignVariableOpdense_4/kernel)dense_4/kernel/Initializer/random_uniform*!
_class
loc:@dense_4/kernel*
dtype0
Х
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
:	Р
М
dense_4/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
:
в
dense_4/biasVarHandleOp*
dtype0*
shape:*
	container *
_class
loc:@dense_4/bias*
shared_namedense_4/bias*
_output_shapes
: 
i
-dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_4/bias*
_output_shapes
: 
Г
dense_4/bias/AssignAssignVariableOpdense_4/biasdense_4/bias/Initializer/zeros*
_class
loc:@dense_4/bias*
dtype0
К
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
:
g
MatMul_4/ReadVariableOpReadVariableOpdense_4/kernel*
dtype0*
_output_shapes
:	Р
О
MatMul_4MatMul	Reshape_2MatMul_4/ReadVariableOp*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:         
a
BiasAdd_4/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:
Б
	BiasAdd_4BiasAddMatMul_4BiasAdd_4/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:         *
T0
K
Relu_2Relu	BiasAdd_4*'
_output_shapes
:         *
T0
f
cond_2/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

M
cond_2/switch_tIdentitycond_2/Switch:1*
_output_shapes
: *
T0

K
cond_2/switch_fIdentitycond_2/Switch*
_output_shapes
: *
T0

Q
cond_2/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
j
cond_2/dropout/rateConst^cond_2/switch_t*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
q
cond_2/dropout/ShapeShapecond_2/dropout/Shape/Switch:1*
out_type0*
_output_shapes
:*
T0
Э
cond_2/dropout/Shape/SwitchSwitchRelu_2cond_2/pred_id*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         *
T0
k
cond_2/dropout/sub/xConst^cond_2/switch_t*
dtype0*
_output_shapes
: *
valueB
 *  А?
e
cond_2/dropout/subSubcond_2/dropout/sub/xcond_2/dropout/rate*
T0*
_output_shapes
: 
x
!cond_2/dropout/random_uniform/minConst^cond_2/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
x
!cond_2/dropout/random_uniform/maxConst^cond_2/switch_t*
valueB
 *  А?*
dtype0*
_output_shapes
: 
к
+cond_2/dropout/random_uniform/RandomUniformRandomUniformcond_2/dropout/Shape*
dtype0*'
_output_shapes
:         *
seed2 *
T0*

seed 
П
!cond_2/dropout/random_uniform/subSub!cond_2/dropout/random_uniform/max!cond_2/dropout/random_uniform/min*
_output_shapes
: *
T0
к
!cond_2/dropout/random_uniform/mulMul+cond_2/dropout/random_uniform/RandomUniform!cond_2/dropout/random_uniform/sub*'
_output_shapes
:         *
T0
Ь
cond_2/dropout/random_uniformAdd!cond_2/dropout/random_uniform/mul!cond_2/dropout/random_uniform/min*'
_output_shapes
:         *
T0
~
cond_2/dropout/addAddcond_2/dropout/subcond_2/dropout/random_uniform*
T0*'
_output_shapes
:         
c
cond_2/dropout/FloorFloorcond_2/dropout/add*'
_output_shapes
:         *
T0
Ж
cond_2/dropout/truedivRealDivcond_2/dropout/Shape/Switch:1cond_2/dropout/sub*'
_output_shapes
:         *
T0
y
cond_2/dropout/mulMulcond_2/dropout/truedivcond_2/dropout/Floor*
T0*'
_output_shapes
:         
e
cond_2/IdentityIdentitycond_2/Identity/Switch*'
_output_shapes
:         *
T0
Ш
cond_2/Identity/SwitchSwitchRelu_2cond_2/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@Relu_2
w
cond_2/MergeMergecond_2/Identitycond_2/dropout/mul*
N*
T0*)
_output_shapes
:         : 
г
/dense_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   *!
_class
loc:@dense_5/kernel
Х
-dense_5/kernel/Initializer/random_uniform/minConst*
valueB
 *ЇЇї╛*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 
Х
-dense_5/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЇЇї>*!
_class
loc:@dense_5/kernel
ы
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:

╓
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_5/kernel
ш
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:

┌
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:

м
dense_5/kernelVarHandleOp*
shared_namedense_5/kernel*
_output_shapes
: *
dtype0*
shape
:
*
	container *!
_class
loc:@dense_5/kernel
m
/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/kernel*
_output_shapes
: 
Ф
dense_5/kernel/AssignAssignVariableOpdense_5/kernel)dense_5/kernel/Initializer/random_uniform*!
_class
loc:@dense_5/kernel*
dtype0
Ф
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes

:

М
dense_5/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:

в
dense_5/biasVarHandleOp*
shape:
*
	container *
_class
loc:@dense_5/bias*
_output_shapes
: *
shared_namedense_5/bias*
dtype0
i
-dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_5/bias*
_output_shapes
: 
Г
dense_5/bias/AssignAssignVariableOpdense_5/biasdense_5/bias/Initializer/zeros*
_class
loc:@dense_5/bias*
dtype0
К
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
:

f
MatMul_5/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0*
_output_shapes

:

С
MatMul_5MatMulcond_2/MergeMatMul_5/ReadVariableOp*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:         

a
BiasAdd_5/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:

Б
	BiasAdd_5BiasAddMatMul_5BiasAdd_5/ReadVariableOp*
data_formatNHWC*'
_output_shapes
:         
*
T0
Q
	Softmax_2Softmax	BiasAdd_5*
T0*'
_output_shapes
:         

Ж
output_1_target_2Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
T
Const_2Const*
dtype0*
_output_shapes
:*
valueB*  А?
И
output_1_sample_weights_2PlaceholderWithDefaultConst_2*
shape:         *
dtype0*#
_output_shapes
:         
z
total_2/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@total_2*
dtype0
П
total_2VarHandleOp*
dtype0*
	container *
shape: *
_class
loc:@total_2*
_output_shapes
: *
shared_name	total_2
_
(total_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal_2*
_output_shapes
: 
o
total_2/AssignAssignVariableOptotal_2total_2/Initializer/zeros*
_class
loc:@total_2*
dtype0
w
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_class
loc:@total_2*
dtype0*
_output_shapes
: 
z
count_2/Initializer/zerosConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@count_2*
dtype0
П
count_2VarHandleOp*
dtype0*
shape: *
	container *
_class
loc:@count_2*
_output_shapes
: *
shared_name	count_2
_
(count_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount_2*
_output_shapes
: 
o
count_2/AssignAssignVariableOpcount_2count_2/Initializer/zeros*
_class
loc:@count_2*
dtype0
w
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_class
loc:@count_2*
dtype0*
_output_shapes
: 
u
"loss_2/output_1_loss/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ъ
loss_2/output_1_loss/ReshapeReshapeoutput_1_target_2"loss_2/output_1_loss/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0
М
loss_2/output_1_loss/CastCastloss_2/output_1_loss/Reshape*

SrcT0*
Truncate( *

DstT0	*#
_output_shapes
:         
u
$loss_2/output_1_loss/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"    
   
Ъ
loss_2/output_1_loss/Reshape_1Reshape	BiasAdd_5$loss_2/output_1_loss/Reshape_1/shape*'
_output_shapes
:         
*
T0*
Tshape0
Ч
>loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_2/output_1_loss/Cast*
T0	*
out_type0*
_output_shapes
:
О
\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_2/output_1_loss/Reshape_1loss_2/output_1_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":         :         

в
Iloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeShapeoutput_1_sample_weights_2*
T0*
out_type0*
_output_shapes
:
К
Hloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
ф
Hloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*
_output_shapes
:*
T0
Й
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
Й
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xConst*
dtype0*
_output_shapes
: *
value	B : 
В
Eloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEqualGloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar/xHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
М
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
╒
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentitySloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

╙
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentityQloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
_output_shapes
: *
T0

╞
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: *
T0

ї
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0
*X
_classN
LJloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
С
qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualxloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchzloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Ю
xloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchGloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rankRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*Z
_classP
NLloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rank
в
zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rankRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 
■
kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

Й
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitymloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

З
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitykloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

М
lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: *
T0

└
Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
┘
Аloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЛloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
╕
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id* 
_output_shapes
::*
T0*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
Ф
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЗloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchlloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id* 
_output_shapes
::*
T0*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape
╟
Еloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
:*
valueB"      *
dtype0
╕
Еloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╥
loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЕloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЕloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
_output_shapes

:*
T0*

index_type0
┤
Бloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╬
|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Аloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeБloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
_output_shapes

:*
N*
T0
┬
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
_output_shapes
: *
valueB :
         *
dtype0
▀
Вloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsНloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*

Tdim0*
_output_shapes

:
╝
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchIloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeRloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Щ
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЙloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchlloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
е
Оloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationВloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
set_operationa-b*<
_output_shapes*
(:         :         :*
validate_indices(*
T0
╙
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeРloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
й
wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstn^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
dtype0*
_output_shapes
: *
value	B : 
б
uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualwloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЖloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
_output_shapes
: *
T0
В
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchqloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranklloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Д
_classz
xvloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Е
jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergemloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
N*
T0

╚
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergejloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
_output_shapes
: : *
N*
T0

й
Aloss_2/output_1_loss/broadcast_weights/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Т
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Я
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_2Const*,
value#B! Boutput_1_sample_weights_2:0*
dtype0*
_output_shapes
: 
С
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
т
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_4Const*o
valuefBd B^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0*
dtype0*
_output_shapes
: 
О
Closs_2/output_1_loss/broadcast_weights/assert_broadcastable/Const_5Const*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
Я
Nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergePloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

╧
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

═
Ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityNloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
╬
Oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
з
Lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOpQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
Н
Zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_tM^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
Р
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.
ў
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Д
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*,
value#B! Boutput_1_sample_weights_2:0*
dtype0*
_output_shapes
: 
Ў
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
╟
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
dtype0*
_output_shapes
: *o
valuefBd B^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0
є
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7ConstQ^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
щ
Nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssertUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchUloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
К
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/MergeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
Ж
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchIloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shapeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*\
_classR
PNloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape* 
_output_shapes
::*
T0
Д
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchHloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shapeOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*[
_classQ
OMloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape* 
_output_shapes
::*
T0
Ў
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchEloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalarOloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*X
_classN
LJloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

С
\loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityPloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_fO^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*c
_classY
WUloc:@loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╝
Mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/MergeMerge\loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1Zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
_output_shapes
: : *
N*
T0

в
6loss_2/output_1_loss/broadcast_weights/ones_like/ShapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsN^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
╦
6loss_2/output_1_loss/broadcast_weights/ones_like/ConstConstN^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
valueB
 *  А?*
dtype0
ш
0loss_2/output_1_loss/broadcast_weights/ones_likeFill6loss_2/output_1_loss/broadcast_weights/ones_like/Shape6loss_2/output_1_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:         *
T0*

index_type0
и
&loss_2/output_1_loss/broadcast_weightsMuloutput_1_sample_weights_20loss_2/output_1_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
╙
loss_2/output_1_loss/MulMul\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits&loss_2/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0
d
loss_2/output_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
У
loss_2/output_1_loss/SumSumloss_2/output_1_loss/Mulloss_2/output_1_loss/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
f
loss_2/output_1_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
е
loss_2/output_1_loss/Sum_1Sum&loss_2/output_1_loss/broadcast_weightsloss_2/output_1_loss/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
В
loss_2/output_1_loss/div_no_nanDivNoNanloss_2/output_1_loss/Sumloss_2/output_1_loss/Sum_1*
_output_shapes
: *
T0
_
loss_2/output_1_loss/Const_2Const*
_output_shapes
: *
valueB *
dtype0
Ю
loss_2/output_1_loss/MeanMeanloss_2/output_1_loss/div_no_nanloss_2/output_1_loss/Const_2*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
Q
loss_2/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
[

loss_2/mulMulloss_2/mul/xloss_2/output_1_loss/Mean*
_output_shapes
: *
T0
З
metrics_2/acc/CastCastoutput_1_target_2*

SrcT0*
Truncate( *

DstT0*0
_output_shapes
:                  
В
metrics_2/acc/SqueezeSqueezemetrics_2/acc/Cast*#
_output_shapes
:         *
T0*
squeeze_dims

         
i
metrics_2/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ц
metrics_2/acc/ArgMaxArgMax	Softmax_2metrics_2/acc/ArgMax/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:         

metrics_2/acc/Cast_1Castmetrics_2/acc/ArgMax*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:         
w
metrics_2/acc/EqualEqualmetrics_2/acc/Squeezemetrics_2/acc/Cast_1*#
_output_shapes
:         *
T0
~
metrics_2/acc/Cast_2Castmetrics_2/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:         
a
metrics_2/acc/SizeSizemetrics_2/acc/Cast_2*
_output_shapes
: *
T0*
out_type0
p
metrics_2/acc/Cast_3Castmetrics_2/acc/Size*

DstT0*
_output_shapes
: *

SrcT0*
Truncate( 
]
metrics_2/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Б
metrics_2/acc/SumSummetrics_2/acc/Cast_2metrics_2/acc/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
a
!metrics_2/acc/AssignAddVariableOpAssignAddVariableOptotal_2metrics_2/acc/Sum*
dtype0
А
metrics_2/acc/ReadVariableOpReadVariableOptotal_2"^metrics_2/acc/AssignAddVariableOp*
dtype0*
_output_shapes
: 
Е
#metrics_2/acc/AssignAddVariableOp_1AssignAddVariableOpcount_2metrics_2/acc/Cast_3^metrics_2/acc/ReadVariableOp*
dtype0
г
metrics_2/acc/ReadVariableOp_1ReadVariableOpcount_2$^metrics_2/acc/AssignAddVariableOp_1^metrics_2/acc/ReadVariableOp*
dtype0*
_output_shapes
: 
И
'metrics_2/acc/div_no_nan/ReadVariableOpReadVariableOptotal_2^metrics_2/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
К
)metrics_2/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount_2^metrics_2/acc/ReadVariableOp_1*
dtype0*
_output_shapes
: 
Щ
metrics_2/acc/div_no_nanDivNoNan'metrics_2/acc/div_no_nan/ReadVariableOp)metrics_2/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Г
metrics_2/acc/Squeeze_1Squeezeoutput_1_target_2*
squeeze_dims

         *#
_output_shapes
:         *
T0
k
 metrics_2/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ъ
metrics_2/acc/ArgMax_1ArgMax	Softmax_2 metrics_2/acc/ArgMax_1/dimension*
T0*
output_type0	*

Tidx0*#
_output_shapes
:         
Б
metrics_2/acc/Cast_4Castmetrics_2/acc/ArgMax_1*#
_output_shapes
:         *

SrcT0	*
Truncate( *

DstT0
{
metrics_2/acc/Equal_1Equalmetrics_2/acc/Squeeze_1metrics_2/acc/Cast_4*#
_output_shapes
:         *
T0
А
metrics_2/acc/Cast_5Castmetrics_2/acc/Equal_1*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:         
_
metrics_2/acc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Е
metrics_2/acc/MeanMeanmetrics_2/acc/Cast_5metrics_2/acc/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
Б
training_4/Adam/gradients/ShapeConst*
_class
loc:@loss_2/mul*
dtype0*
_output_shapes
: *
valueB 
З
#training_4/Adam/gradients/grad_ys_0Const*
valueB
 *  А?*
_class
loc:@loss_2/mul*
dtype0*
_output_shapes
: 
╛
training_4/Adam/gradients/FillFilltraining_4/Adam/gradients/Shape#training_4/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss_2/mul*
_output_shapes
: 
п
-training_4/Adam/gradients/loss_2/mul_grad/MulMultraining_4/Adam/gradients/Fillloss_2/output_1_loss/Mean*
_output_shapes
: *
T0*
_class
loc:@loss_2/mul
д
/training_4/Adam/gradients/loss_2/mul_grad/Mul_1Multraining_4/Adam/gradients/Fillloss_2/mul/x*
_class
loc:@loss_2/mul*
_output_shapes
: *
T0
╖
Ftraining_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape/shapeConst*
valueB *,
_class"
 loc:@loss_2/output_1_loss/Mean*
dtype0*
_output_shapes
: 
б
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/ReshapeReshape/training_4/Adam/gradients/loss_2/mul_grad/Mul_1Ftraining_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape/shape*
_output_shapes
: *
T0*,
_class"
 loc:@loss_2/output_1_loss/Mean*
Tshape0
п
>training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB *,
_class"
 loc:@loss_2/output_1_loss/Mean
и
=training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/TileTile@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Reshape>training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const*
T0*

Tmultiples0*,
_class"
 loc:@loss_2/output_1_loss/Mean*
_output_shapes
: 
│
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const_1Const*,
_class"
 loc:@loss_2/output_1_loss/Mean*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ы
@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivRealDiv=training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Tile@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/Const_1*,
_class"
 loc:@loss_2/output_1_loss/Mean*
_output_shapes
: *
T0
╗
Dtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ShapeConst*
_output_shapes
: *
valueB *2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
dtype0
╜
Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1Const*
valueB *2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
dtype0*
_output_shapes
: 
ь
Ttraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ShapeFtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*2
_output_shapes 
:         :         *
T0
И
Itraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nanDivNoNan@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivloss_2/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: 
▄
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/SumSumItraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nanTtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
╛
Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ReshapeReshapeBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/SumDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: *
T0
╕
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/NegNegloss_2/output_1_loss/Sum*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *
T0
М
Ktraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_1DivNoNanBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Negloss_2/output_1_loss/Sum_1*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *
T0
Х
Ktraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_2DivNoNanKtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_1loss_2/output_1_loss/Sum_1*
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: 
н
Btraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/mulMul@training_4/Adam/gradients/loss_2/output_1_loss/Mean_grad/truedivKtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/div_no_nan_2*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
_output_shapes
: *
T0
┘
Dtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Sum_1SumBtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/mulVtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan
─
Htraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Reshape_1ReshapeDtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Sum_1Ftraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/Shape_1*
T0*2
_class(
&$loc:@loss_2/output_1_loss/div_no_nan*
Tshape0*
_output_shapes
: 
╝
Etraining_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss_2/output_1_loss/Sum*
dtype0*
_output_shapes
:
╣
?training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/ReshapeReshapeFtraining_4/Adam/gradients/loss_2/output_1_loss/div_no_nan_grad/ReshapeEtraining_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape/shape*
T0*+
_class!
loc:@loss_2/output_1_loss/Sum*
Tshape0*
_output_shapes
:
┬
=training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/ShapeShapeloss_2/output_1_loss/Mul*
T0*
out_type0*+
_class!
loc:@loss_2/output_1_loss/Sum*
_output_shapes
:
▒
<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/TileTile?training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Reshape=training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Shape*
T0*

Tmultiples0*+
_class!
loc:@loss_2/output_1_loss/Sum*#
_output_shapes
:         
Ж
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ShapeShape\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*+
_class!
loc:@loss_2/output_1_loss/Mul*
_output_shapes
:
╥
?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1Shape&loss_2/output_1_loss/broadcast_weights*
T0*
out_type0*+
_class!
loc:@loss_2/output_1_loss/Mul*
_output_shapes
:
╨
Mtraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1*
T0*+
_class!
loc:@loss_2/output_1_loss/Mul*2
_output_shapes 
:         :         
Г
;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/MulMul<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Tile&loss_2/output_1_loss/broadcast_weights*#
_output_shapes
:         *
T0*+
_class!
loc:@loss_2/output_1_loss/Mul
╗
;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/SumSum;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/MulMtraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgs*+
_class!
loc:@loss_2/output_1_loss/Mul*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
п
?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ReshapeReshape;training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape*+
_class!
loc:@loss_2/output_1_loss/Mul*
Tshape0*#
_output_shapes
:         *
T0
╗
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Mul_1Mul\loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits<training_4/Adam/gradients/loss_2/output_1_loss/Sum_grad/Tile*+
_class!
loc:@loss_2/output_1_loss/Mul*#
_output_shapes
:         *
T0
┴
=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum_1Sum=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Mul_1Otraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/BroadcastGradientArgs:1*+
_class!
loc:@loss_2/output_1_loss/Mul*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
╡
Atraining_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Reshape_1Reshape=training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Sum_1?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/Shape_1*
T0*+
_class!
loc:@loss_2/output_1_loss/Mul*
Tshape0*#
_output_shapes
:         
┤
$training_4/Adam/gradients/zeros_like	ZerosLike^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
*
T0
┘
Лtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient^loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*┤
messageиеCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         

╟
Кtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0*
_output_shapes
: 
Т
Жtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims?training_4/Adam/gradients/loss_2/output_1_loss/Mul_grad/ReshapeКtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
└
training_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulЖtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsЛtraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*o
_classe
caloc:@loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:         
*
T0
┐
Ctraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ShapeShape	BiasAdd_5*
T0*
out_type0*1
_class'
%#loc:@loss_2/output_1_loss/Reshape_1*
_output_shapes
:
Й
Etraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ReshapeReshapetraining_4/Adam/gradients/loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulCtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Shape*1
_class'
%#loc:@loss_2/output_1_loss/Reshape_1*
Tshape0*'
_output_shapes
:         
*
T0
ф
4training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGradBiasAddGradEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@BiasAdd_5*
_output_shapes
:
*
T0*
data_formatNHWC
Н
.training_4/Adam/gradients/MatMul_5_grad/MatMulMatMulEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/ReshapeMatMul_5/ReadVariableOp*
transpose_a( *
T0*
_class
loc:@MatMul_5*
transpose_b(*'
_output_shapes
:         
√
0training_4/Adam/gradients/MatMul_5_grad/MatMul_1MatMulcond_2/MergeEtraining_4/Adam/gradients/loss_2/output_1_loss/Reshape_1_grad/Reshape*
_class
loc:@MatMul_5*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
с
5training_4/Adam/gradients/cond_2/Merge_grad/cond_gradSwitch.training_4/Adam/gradients/MatMul_5_grad/MatMulcond_2/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@MatMul_5
┤
7training_4/Adam/gradients/cond_2/dropout/mul_grad/ShapeShapecond_2/dropout/truediv*
_output_shapes
:*
T0*
out_type0*%
_class
loc:@cond_2/dropout/mul
┤
9training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1Shapecond_2/dropout/Floor*%
_class
loc:@cond_2/dropout/mul*
_output_shapes
:*
T0*
out_type0
╕
Gtraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs7training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape9training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*%
_class
loc:@cond_2/dropout/mul*2
_output_shapes 
:         :         *
T0
ф
5training_4/Adam/gradients/cond_2/dropout/mul_grad/MulMul7training_4/Adam/gradients/cond_2/Merge_grad/cond_grad:1cond_2/dropout/Floor*%
_class
loc:@cond_2/dropout/mul*'
_output_shapes
:         *
T0
г
5training_4/Adam/gradients/cond_2/dropout/mul_grad/SumSum5training_4/Adam/gradients/cond_2/dropout/mul_grad/MulGtraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs*
T0*%
_class
loc:@cond_2/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:
Ы
9training_4/Adam/gradients/cond_2/dropout/mul_grad/ReshapeReshape5training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum7training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape*'
_output_shapes
:         *
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0
ш
7training_4/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Mulcond_2/dropout/truediv7training_4/Adam/gradients/cond_2/Merge_grad/cond_grad:1*'
_output_shapes
:         *
T0*%
_class
loc:@cond_2/dropout/mul
й
7training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum_1Sum7training_4/Adam/gradients/cond_2/dropout/mul_grad/Mul_1Itraining_4/Adam/gradients/cond_2/dropout/mul_grad/BroadcastGradientArgs:1*%
_class
loc:@cond_2/dropout/mul*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
б
;training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshape_1Reshape7training_4/Adam/gradients/cond_2/dropout/mul_grad/Sum_19training_4/Adam/gradients/cond_2/dropout/mul_grad/Shape_1*
T0*%
_class
loc:@cond_2/dropout/mul*
Tshape0*'
_output_shapes
:         
в
 training_4/Adam/gradients/SwitchSwitchRelu_2cond_2/pred_id*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         *
T0
Я
"training_4/Adam/gradients/IdentityIdentity"training_4/Adam/gradients/Switch:1*
T0*
_class
loc:@Relu_2*'
_output_shapes
:         
Ю
!training_4/Adam/gradients/Shape_1Shape"training_4/Adam/gradients/Switch:1*
_class
loc:@Relu_2*
_output_shapes
:*
T0*
out_type0
к
%training_4/Adam/gradients/zeros/ConstConst#^training_4/Adam/gradients/Identity*
valueB
 *    *
_class
loc:@Relu_2*
dtype0*
_output_shapes
: 
╨
training_4/Adam/gradients/zerosFill!training_4/Adam/gradients/Shape_1%training_4/Adam/gradients/zeros/Const*'
_output_shapes
:         *
T0*

index_type0*
_class
loc:@Relu_2
°
?training_4/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradMerge5training_4/Adam/gradients/cond_2/Merge_grad/cond_gradtraining_4/Adam/gradients/zeros*
_class
loc:@Relu_2*)
_output_shapes
:         : *
T0*
N
├
;training_4/Adam/gradients/cond_2/dropout/truediv_grad/ShapeShapecond_2/dropout/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@cond_2/dropout/truediv
л
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *)
_class
loc:@cond_2/dropout/truediv
╚
Ktraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1*)
_class
loc:@cond_2/dropout/truediv*2
_output_shapes 
:         :         *
T0
Ї
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDivRealDiv9training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshapecond_2/dropout/sub*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         *
T0
╖
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/SumSum=training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDivKtraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgs*
T0*)
_class
loc:@cond_2/dropout/truediv*
	keep_dims( *

Tidx0*
_output_shapes
:
л
=training_4/Adam/gradients/cond_2/dropout/truediv_grad/ReshapeReshape9training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape*)
_class
loc:@cond_2/dropout/truediv*
Tshape0*'
_output_shapes
:         *
T0
╝
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/NegNegcond_2/dropout/Shape/Switch:1*'
_output_shapes
:         *
T0*)
_class
loc:@cond_2/dropout/truediv
Ў
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_1RealDiv9training_4/Adam/gradients/cond_2/dropout/truediv_grad/Negcond_2/dropout/sub*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         *
T0
№
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_2RealDiv?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_1cond_2/dropout/sub*)
_class
loc:@cond_2/dropout/truediv*'
_output_shapes
:         *
T0
Щ
9training_4/Adam/gradients/cond_2/dropout/truediv_grad/mulMul9training_4/Adam/gradients/cond_2/dropout/mul_grad/Reshape?training_4/Adam/gradients/cond_2/dropout/truediv_grad/RealDiv_2*'
_output_shapes
:         *
T0*)
_class
loc:@cond_2/dropout/truediv
╖
;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum_1Sum9training_4/Adam/gradients/cond_2/dropout/truediv_grad/mulMtraining_4/Adam/gradients/cond_2/dropout/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0*)
_class
loc:@cond_2/dropout/truediv
а
?training_4/Adam/gradients/cond_2/dropout/truediv_grad/Reshape_1Reshape;training_4/Adam/gradients/cond_2/dropout/truediv_grad/Sum_1=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Shape_1*
T0*)
_class
loc:@cond_2/dropout/truediv*
Tshape0*
_output_shapes
: 
д
"training_4/Adam/gradients/Switch_1SwitchRelu_2cond_2/pred_id*
T0*
_class
loc:@Relu_2*:
_output_shapes(
&:         :         
б
$training_4/Adam/gradients/Identity_1Identity"training_4/Adam/gradients/Switch_1*
_class
loc:@Relu_2*'
_output_shapes
:         *
T0
Ю
!training_4/Adam/gradients/Shape_2Shape"training_4/Adam/gradients/Switch_1*
T0*
out_type0*
_class
loc:@Relu_2*
_output_shapes
:
о
'training_4/Adam/gradients/zeros_1/ConstConst%^training_4/Adam/gradients/Identity_1*
_class
loc:@Relu_2*
dtype0*
_output_shapes
: *
valueB
 *    
╘
!training_4/Adam/gradients/zeros_1Fill!training_4/Adam/gradients/Shape_2'training_4/Adam/gradients/zeros_1/Const*
_class
loc:@Relu_2*'
_output_shapes
:         *
T0*

index_type0
З
Dtraining_4/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_gradMerge!training_4/Adam/gradients/zeros_1=training_4/Adam/gradients/cond_2/dropout/truediv_grad/Reshape*)
_output_shapes
:         : *
T0*
N*
_class
loc:@Relu_2
Г
training_4/Adam/gradients/AddNAddN?training_4/Adam/gradients/cond_2/Identity/Switch_grad/cond_gradDtraining_4/Adam/gradients/cond_2/dropout/Shape/Switch_grad/cond_grad*'
_output_shapes
:         *
T0*
N*
_class
loc:@Relu_2
п
.training_4/Adam/gradients/Relu_2_grad/ReluGradReluGradtraining_4/Adam/gradients/AddNRelu_2*
_class
loc:@Relu_2*'
_output_shapes
:         *
T0
═
4training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad.training_4/Adam/gradients/Relu_2_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC*
_class
loc:@BiasAdd_4
ў
.training_4/Adam/gradients/MatMul_4_grad/MatMulMatMul.training_4/Adam/gradients/Relu_2_grad/ReluGradMatMul_4/ReadVariableOp*
_class
loc:@MatMul_4*
transpose_b(*(
_output_shapes
:         Р*
transpose_a( *
T0
т
0training_4/Adam/gradients/MatMul_4_grad/MatMul_1MatMul	Reshape_2.training_4/Adam/gradients/Relu_2_grad/ReluGrad*
transpose_a(*
T0*
_class
loc:@MatMul_4*
transpose_b( *
_output_shapes
:	Р
W
training_4/Adam/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
q
#training_4/Adam/AssignAddVariableOpAssignAddVariableOpAdam_1/iterationstraining_4/Adam/Const*
dtype0	
О
training_4/Adam/ReadVariableOpReadVariableOpAdam_1/iterations$^training_4/Adam/AssignAddVariableOp*
_output_shapes
: *
dtype0	
О
#training_4/Adam/Cast/ReadVariableOpReadVariableOpAdam_1/iterations^training_4/Adam/ReadVariableOp*
_output_shapes
: *
dtype0	
Б
training_4/Adam/CastCast#training_4/Adam/Cast/ReadVariableOp*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
h
"training_4/Adam/Pow/ReadVariableOpReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
u
training_4/Adam/PowPow"training_4/Adam/Pow/ReadVariableOptraining_4/Adam/Cast*
T0*
_output_shapes
: 
Z
training_4/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
training_4/Adam/subSubtraining_4/Adam/sub/xtraining_4/Adam/Pow*
T0*
_output_shapes
: 
\
training_4/Adam/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_2Const*
valueB
 *  А*
dtype0*
_output_shapes
: 

%training_4/Adam/clip_by_value/MinimumMinimumtraining_4/Adam/subtraining_4/Adam/Const_2*
_output_shapes
: *
T0
Й
training_4/Adam/clip_by_valueMaximum%training_4/Adam/clip_by_value/Minimumtraining_4/Adam/Const_1*
_output_shapes
: *
T0
\
training_4/Adam/SqrtSqrttraining_4/Adam/clip_by_value*
T0*
_output_shapes
: 
j
$training_4/Adam/Pow_1/ReadVariableOpReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
y
training_4/Adam/Pow_1Pow$training_4/Adam/Pow_1/ReadVariableOptraining_4/Adam/Cast*
T0*
_output_shapes
: 
\
training_4/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
m
training_4/Adam/sub_1Subtraining_4/Adam/sub_1/xtraining_4/Adam/Pow_1*
T0*
_output_shapes
: 
p
training_4/Adam/truedivRealDivtraining_4/Adam/Sqrttraining_4/Adam/sub_1*
_output_shapes
: *
T0
b
 training_4/Adam/ReadVariableOp_1ReadVariableOp	Adam_1/lr*
dtype0*
_output_shapes
: 
v
training_4/Adam/mulMul training_4/Adam/ReadVariableOp_1training_4/Adam/truediv*
_output_shapes
: *
T0
v
%training_4/Adam/zeros/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
`
training_4/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
training_4/Adam/zerosFill%training_4/Adam/zeros/shape_as_tensortraining_4/Adam/zeros/Const*
_output_shapes
:	Р*
T0*

index_type0
╦
training_4/Adam/VariableVarHandleOp*
shape:	Р*
	container *+
_class!
loc:@training_4/Adam/Variable*)
shared_nametraining_4/Adam/Variable*
_output_shapes
: *
dtype0
Б
9training_4/Adam/Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable*
_output_shapes
: 
Ю
training_4/Adam/Variable/AssignAssignVariableOptraining_4/Adam/Variabletraining_4/Adam/zeros*+
_class!
loc:@training_4/Adam/Variable*
dtype0
│
,training_4/Adam/Variable/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable*+
_class!
loc:@training_4/Adam/Variable*
dtype0*
_output_shapes
:	Р
d
training_4/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
╠
training_4/Adam/Variable_1VarHandleOp*
dtype0*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_1*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_1
Е
;training_4/Adam/Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_1*
_output_shapes
: 
ж
!training_4/Adam/Variable_1/AssignAssignVariableOptraining_4/Adam/Variable_1training_4/Adam/zeros_1*-
_class#
!loc:@training_4/Adam/Variable_1*
dtype0
┤
.training_4/Adam/Variable_1/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_1*-
_class#
!loc:@training_4/Adam/Variable_1*
dtype0*
_output_shapes
:
l
training_4/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

╨
training_4/Adam/Variable_2VarHandleOp*-
_class#
!loc:@training_4/Adam/Variable_2*+
shared_nametraining_4/Adam/Variable_2*
_output_shapes
: *
dtype0*
shape
:
*
	container 
Е
;training_4/Adam/Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_2*
_output_shapes
: 
ж
!training_4/Adam/Variable_2/AssignAssignVariableOptraining_4/Adam/Variable_2training_4/Adam/zeros_2*-
_class#
!loc:@training_4/Adam/Variable_2*
dtype0
╕
.training_4/Adam/Variable_2/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_2*-
_class#
!loc:@training_4/Adam/Variable_2*
dtype0*
_output_shapes

:

d
training_4/Adam/zeros_3Const*
dtype0*
_output_shapes
:
*
valueB
*    
╠
training_4/Adam/Variable_3VarHandleOp*
shape:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_3*+
shared_nametraining_4/Adam/Variable_3*
_output_shapes
: *
dtype0
Е
;training_4/Adam/Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_3*
_output_shapes
: 
ж
!training_4/Adam/Variable_3/AssignAssignVariableOptraining_4/Adam/Variable_3training_4/Adam/zeros_3*-
_class#
!loc:@training_4/Adam/Variable_3*
dtype0
┤
.training_4/Adam/Variable_3/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_3*
_output_shapes
:
*-
_class#
!loc:@training_4/Adam/Variable_3*
dtype0
x
'training_4/Adam/zeros_4/shape_as_tensorConst*
valueB"     *
dtype0*
_output_shapes
:
b
training_4/Adam/zeros_4/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
г
training_4/Adam/zeros_4Fill'training_4/Adam/zeros_4/shape_as_tensortraining_4/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes
:	Р
╤
training_4/Adam/Variable_4VarHandleOp*
dtype0*
shape:	Р*
	container *-
_class#
!loc:@training_4/Adam/Variable_4*+
shared_nametraining_4/Adam/Variable_4*
_output_shapes
: 
Е
;training_4/Adam/Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_4*
_output_shapes
: 
ж
!training_4/Adam/Variable_4/AssignAssignVariableOptraining_4/Adam/Variable_4training_4/Adam/zeros_4*-
_class#
!loc:@training_4/Adam/Variable_4*
dtype0
╣
.training_4/Adam/Variable_4/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_4*-
_class#
!loc:@training_4/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
d
training_4/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
╠
training_4/Adam/Variable_5VarHandleOp*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_5*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_5*
dtype0
Е
;training_4/Adam/Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_5*
_output_shapes
: 
ж
!training_4/Adam/Variable_5/AssignAssignVariableOptraining_4/Adam/Variable_5training_4/Adam/zeros_5*-
_class#
!loc:@training_4/Adam/Variable_5*
dtype0
┤
.training_4/Adam/Variable_5/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_5*-
_class#
!loc:@training_4/Adam/Variable_5*
dtype0*
_output_shapes
:
l
training_4/Adam/zeros_6Const*
_output_shapes

:
*
valueB
*    *
dtype0
╨
training_4/Adam/Variable_6VarHandleOp*
dtype0*
shape
:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_6*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_6
Е
;training_4/Adam/Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_6*
_output_shapes
: 
ж
!training_4/Adam/Variable_6/AssignAssignVariableOptraining_4/Adam/Variable_6training_4/Adam/zeros_6*-
_class#
!loc:@training_4/Adam/Variable_6*
dtype0
╕
.training_4/Adam/Variable_6/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_6*
dtype0*
_output_shapes

:
*-
_class#
!loc:@training_4/Adam/Variable_6
d
training_4/Adam/zeros_7Const*
valueB
*    *
dtype0*
_output_shapes
:

╠
training_4/Adam/Variable_7VarHandleOp*
dtype0*
shape:
*
	container *-
_class#
!loc:@training_4/Adam/Variable_7*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_7
Е
;training_4/Adam/Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_7*
_output_shapes
: 
ж
!training_4/Adam/Variable_7/AssignAssignVariableOptraining_4/Adam/Variable_7training_4/Adam/zeros_7*-
_class#
!loc:@training_4/Adam/Variable_7*
dtype0
┤
.training_4/Adam/Variable_7/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_7*-
_class#
!loc:@training_4/Adam/Variable_7*
dtype0*
_output_shapes
:

q
'training_4/Adam/zeros_8/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
b
training_4/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
training_4/Adam/zeros_8Fill'training_4/Adam/zeros_8/shape_as_tensortraining_4/Adam/zeros_8/Const*
_output_shapes
:*
T0*

index_type0
╠
training_4/Adam/Variable_8VarHandleOp*
	container *
shape:*-
_class#
!loc:@training_4/Adam/Variable_8*
_output_shapes
: *+
shared_nametraining_4/Adam/Variable_8*
dtype0
Е
;training_4/Adam/Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_8*
_output_shapes
: 
ж
!training_4/Adam/Variable_8/AssignAssignVariableOptraining_4/Adam/Variable_8training_4/Adam/zeros_8*-
_class#
!loc:@training_4/Adam/Variable_8*
dtype0
┤
.training_4/Adam/Variable_8/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_8*-
_class#
!loc:@training_4/Adam/Variable_8*
dtype0*
_output_shapes
:
q
'training_4/Adam/zeros_9/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
b
training_4/Adam/zeros_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
training_4/Adam/zeros_9Fill'training_4/Adam/zeros_9/shape_as_tensortraining_4/Adam/zeros_9/Const*

index_type0*
_output_shapes
:*
T0
╠
training_4/Adam/Variable_9VarHandleOp*
shape:*
	container *-
_class#
!loc:@training_4/Adam/Variable_9*+
shared_nametraining_4/Adam/Variable_9*
_output_shapes
: *
dtype0
Е
;training_4/Adam/Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_9*
_output_shapes
: 
ж
!training_4/Adam/Variable_9/AssignAssignVariableOptraining_4/Adam/Variable_9training_4/Adam/zeros_9*
dtype0*-
_class#
!loc:@training_4/Adam/Variable_9
┤
.training_4/Adam/Variable_9/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_9*-
_class#
!loc:@training_4/Adam/Variable_9*
dtype0*
_output_shapes
:
r
(training_4/Adam/zeros_10/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
c
training_4/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
б
training_4/Adam/zeros_10Fill(training_4/Adam/zeros_10/shape_as_tensortraining_4/Adam/zeros_10/Const*
T0*

index_type0*
_output_shapes
:
╧
training_4/Adam/Variable_10VarHandleOp*
_output_shapes
: *,
shared_nametraining_4/Adam/Variable_10*
dtype0*
shape:*
	container *.
_class$
" loc:@training_4/Adam/Variable_10
З
<training_4/Adam/Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_10*
_output_shapes
: 
к
"training_4/Adam/Variable_10/AssignAssignVariableOptraining_4/Adam/Variable_10training_4/Adam/zeros_10*.
_class$
" loc:@training_4/Adam/Variable_10*
dtype0
╖
/training_4/Adam/Variable_10/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_10*.
_class$
" loc:@training_4/Adam/Variable_10*
dtype0*
_output_shapes
:
r
(training_4/Adam/zeros_11/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
c
training_4/Adam/zeros_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
б
training_4/Adam/zeros_11Fill(training_4/Adam/zeros_11/shape_as_tensortraining_4/Adam/zeros_11/Const*
_output_shapes
:*
T0*

index_type0
╧
training_4/Adam/Variable_11VarHandleOp*
dtype0*
shape:*
	container *.
_class$
" loc:@training_4/Adam/Variable_11*,
shared_nametraining_4/Adam/Variable_11*
_output_shapes
: 
З
<training_4/Adam/Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining_4/Adam/Variable_11*
_output_shapes
: 
к
"training_4/Adam/Variable_11/AssignAssignVariableOptraining_4/Adam/Variable_11training_4/Adam/zeros_11*.
_class$
" loc:@training_4/Adam/Variable_11*
dtype0
╖
/training_4/Adam/Variable_11/Read/ReadVariableOpReadVariableOptraining_4/Adam/Variable_11*.
_class$
" loc:@training_4/Adam/Variable_11*
dtype0*
_output_shapes
:
f
 training_4/Adam/ReadVariableOp_2ReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
~
$training_4/Adam/mul_1/ReadVariableOpReadVariableOptraining_4/Adam/Variable*
dtype0*
_output_shapes
:	Р
О
training_4/Adam/mul_1Mul training_4/Adam/ReadVariableOp_2$training_4/Adam/mul_1/ReadVariableOp*
T0*
_output_shapes
:	Р
f
 training_4/Adam/ReadVariableOp_3ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
x
training_4/Adam/sub_2Subtraining_4/Adam/sub_2/x training_4/Adam/ReadVariableOp_3*
T0*
_output_shapes
: 
П
training_4/Adam/mul_2Multraining_4/Adam/sub_20training_4/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes
:	Р*
T0
r
training_4/Adam/addAddtraining_4/Adam/mul_1training_4/Adam/mul_2*
T0*
_output_shapes
:	Р
f
 training_4/Adam/ReadVariableOp_4ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
А
$training_4/Adam/mul_3/ReadVariableOpReadVariableOptraining_4/Adam/Variable_4*
dtype0*
_output_shapes
:	Р
О
training_4/Adam/mul_3Mul training_4/Adam/ReadVariableOp_4$training_4/Adam/mul_3/ReadVariableOp*
_output_shapes
:	Р*
T0
f
 training_4/Adam/ReadVariableOp_5ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
x
training_4/Adam/sub_3Subtraining_4/Adam/sub_3/x training_4/Adam/ReadVariableOp_5*
T0*
_output_shapes
: 
|
training_4/Adam/SquareSquare0training_4/Adam/gradients/MatMul_4_grad/MatMul_1*
_output_shapes
:	Р*
T0
u
training_4/Adam/mul_4Multraining_4/Adam/sub_3training_4/Adam/Square*
_output_shapes
:	Р*
T0
t
training_4/Adam/add_1Addtraining_4/Adam/mul_3training_4/Adam/mul_4*
_output_shapes
:	Р*
T0
p
training_4/Adam/mul_5Multraining_4/Adam/multraining_4/Adam/add*
_output_shapes
:	Р*
T0
\
training_4/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *    
\
training_4/Adam/Const_4Const*
_output_shapes
: *
valueB
 *  А*
dtype0
М
'training_4/Adam/clip_by_value_1/MinimumMinimumtraining_4/Adam/add_1training_4/Adam/Const_4*
T0*
_output_shapes
:	Р
Ц
training_4/Adam/clip_by_value_1Maximum'training_4/Adam/clip_by_value_1/Minimumtraining_4/Adam/Const_3*
_output_shapes
:	Р*
T0
i
training_4/Adam/Sqrt_1Sqrttraining_4/Adam/clip_by_value_1*
T0*
_output_shapes
:	Р
\
training_4/Adam/add_2/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
w
training_4/Adam/add_2Addtraining_4/Adam/Sqrt_1training_4/Adam/add_2/y*
_output_shapes
:	Р*
T0
|
training_4/Adam/truediv_1RealDivtraining_4/Adam/mul_5training_4/Adam/add_2*
_output_shapes
:	Р*
T0
p
 training_4/Adam/ReadVariableOp_6ReadVariableOpdense_4/kernel*
dtype0*
_output_shapes
:	Р
Г
training_4/Adam/sub_4Sub training_4/Adam/ReadVariableOp_6training_4/Adam/truediv_1*
_output_shapes
:	Р*
T0
p
 training_4/Adam/AssignVariableOpAssignVariableOptraining_4/Adam/Variabletraining_4/Adam/add*
dtype0
Э
 training_4/Adam/ReadVariableOp_7ReadVariableOptraining_4/Adam/Variable!^training_4/Adam/AssignVariableOp*
dtype0*
_output_shapes
:	Р
v
"training_4/Adam/AssignVariableOp_1AssignVariableOptraining_4/Adam/Variable_4training_4/Adam/add_1*
dtype0
б
 training_4/Adam/ReadVariableOp_8ReadVariableOptraining_4/Adam/Variable_4#^training_4/Adam/AssignVariableOp_1*
dtype0*
_output_shapes
:	Р
j
"training_4/Adam/AssignVariableOp_2AssignVariableOpdense_4/kerneltraining_4/Adam/sub_4*
dtype0
Х
 training_4/Adam/ReadVariableOp_9ReadVariableOpdense_4/kernel#^training_4/Adam/AssignVariableOp_2*
dtype0*
_output_shapes
:	Р
g
!training_4/Adam/ReadVariableOp_10ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
{
$training_4/Adam/mul_6/ReadVariableOpReadVariableOptraining_4/Adam/Variable_1*
dtype0*
_output_shapes
:
К
training_4/Adam/mul_6Mul!training_4/Adam/ReadVariableOp_10$training_4/Adam/mul_6/ReadVariableOp*
T0*
_output_shapes
:
g
!training_4/Adam/ReadVariableOp_11ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_5Subtraining_4/Adam/sub_5/x!training_4/Adam/ReadVariableOp_11*
_output_shapes
: *
T0
О
training_4/Adam/mul_7Multraining_4/Adam/sub_54training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
T0*
_output_shapes
:
o
training_4/Adam/add_3Addtraining_4/Adam/mul_6training_4/Adam/mul_7*
_output_shapes
:*
T0
g
!training_4/Adam/ReadVariableOp_12ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
{
$training_4/Adam/mul_8/ReadVariableOpReadVariableOptraining_4/Adam/Variable_5*
dtype0*
_output_shapes
:
К
training_4/Adam/mul_8Mul!training_4/Adam/ReadVariableOp_12$training_4/Adam/mul_8/ReadVariableOp*
_output_shapes
:*
T0
g
!training_4/Adam/ReadVariableOp_13ReadVariableOpAdam_1/beta_2*
_output_shapes
: *
dtype0
\
training_4/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_6Subtraining_4/Adam/sub_6/x!training_4/Adam/ReadVariableOp_13*
_output_shapes
: *
T0
}
training_4/Adam/Square_1Square4training_4/Adam/gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:*
T0
r
training_4/Adam/mul_9Multraining_4/Adam/sub_6training_4/Adam/Square_1*
T0*
_output_shapes
:
o
training_4/Adam/add_4Addtraining_4/Adam/mul_8training_4/Adam/mul_9*
_output_shapes
:*
T0
n
training_4/Adam/mul_10Multraining_4/Adam/multraining_4/Adam/add_3*
_output_shapes
:*
T0
\
training_4/Adam/Const_5Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *  А
З
'training_4/Adam/clip_by_value_2/MinimumMinimumtraining_4/Adam/add_4training_4/Adam/Const_6*
T0*
_output_shapes
:
С
training_4/Adam/clip_by_value_2Maximum'training_4/Adam/clip_by_value_2/Minimumtraining_4/Adam/Const_5*
_output_shapes
:*
T0
d
training_4/Adam/Sqrt_2Sqrttraining_4/Adam/clip_by_value_2*
_output_shapes
:*
T0
\
training_4/Adam/add_5/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
r
training_4/Adam/add_5Addtraining_4/Adam/Sqrt_2training_4/Adam/add_5/y*
T0*
_output_shapes
:
x
training_4/Adam/truediv_2RealDivtraining_4/Adam/mul_10training_4/Adam/add_5*
_output_shapes
:*
T0
j
!training_4/Adam/ReadVariableOp_14ReadVariableOpdense_4/bias*
dtype0*
_output_shapes
:

training_4/Adam/sub_7Sub!training_4/Adam/ReadVariableOp_14training_4/Adam/truediv_2*
T0*
_output_shapes
:
v
"training_4/Adam/AssignVariableOp_3AssignVariableOptraining_4/Adam/Variable_1training_4/Adam/add_3*
dtype0
Э
!training_4/Adam/ReadVariableOp_15ReadVariableOptraining_4/Adam/Variable_1#^training_4/Adam/AssignVariableOp_3*
dtype0*
_output_shapes
:
v
"training_4/Adam/AssignVariableOp_4AssignVariableOptraining_4/Adam/Variable_5training_4/Adam/add_4*
dtype0
Э
!training_4/Adam/ReadVariableOp_16ReadVariableOptraining_4/Adam/Variable_5#^training_4/Adam/AssignVariableOp_4*
_output_shapes
:*
dtype0
h
"training_4/Adam/AssignVariableOp_5AssignVariableOpdense_4/biastraining_4/Adam/sub_7*
dtype0
П
!training_4/Adam/ReadVariableOp_17ReadVariableOpdense_4/bias#^training_4/Adam/AssignVariableOp_5*
dtype0*
_output_shapes
:
g
!training_4/Adam/ReadVariableOp_18ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
А
%training_4/Adam/mul_11/ReadVariableOpReadVariableOptraining_4/Adam/Variable_2*
dtype0*
_output_shapes

:

Р
training_4/Adam/mul_11Mul!training_4/Adam/ReadVariableOp_18%training_4/Adam/mul_11/ReadVariableOp*
_output_shapes

:
*
T0
g
!training_4/Adam/ReadVariableOp_19ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_8Subtraining_4/Adam/sub_8/x!training_4/Adam/ReadVariableOp_19*
T0*
_output_shapes
: 
П
training_4/Adam/mul_12Multraining_4/Adam/sub_80training_4/Adam/gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:
*
T0
u
training_4/Adam/add_6Addtraining_4/Adam/mul_11training_4/Adam/mul_12*
T0*
_output_shapes

:

g
!training_4/Adam/ReadVariableOp_20ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
А
%training_4/Adam/mul_13/ReadVariableOpReadVariableOptraining_4/Adam/Variable_6*
dtype0*
_output_shapes

:

Р
training_4/Adam/mul_13Mul!training_4/Adam/ReadVariableOp_20%training_4/Adam/mul_13/ReadVariableOp*
_output_shapes

:
*
T0
g
!training_4/Adam/ReadVariableOp_21ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
\
training_4/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
y
training_4/Adam/sub_9Subtraining_4/Adam/sub_9/x!training_4/Adam/ReadVariableOp_21*
_output_shapes
: *
T0
}
training_4/Adam/Square_2Square0training_4/Adam/gradients/MatMul_5_grad/MatMul_1*
_output_shapes

:
*
T0
w
training_4/Adam/mul_14Multraining_4/Adam/sub_9training_4/Adam/Square_2*
_output_shapes

:
*
T0
u
training_4/Adam/add_7Addtraining_4/Adam/mul_13training_4/Adam/mul_14*
_output_shapes

:
*
T0
r
training_4/Adam/mul_15Multraining_4/Adam/multraining_4/Adam/add_6*
_output_shapes

:
*
T0
\
training_4/Adam/Const_7Const*
valueB
 *    *
dtype0*
_output_shapes
: 
\
training_4/Adam/Const_8Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Л
'training_4/Adam/clip_by_value_3/MinimumMinimumtraining_4/Adam/add_7training_4/Adam/Const_8*
_output_shapes

:
*
T0
Х
training_4/Adam/clip_by_value_3Maximum'training_4/Adam/clip_by_value_3/Minimumtraining_4/Adam/Const_7*
T0*
_output_shapes

:

h
training_4/Adam/Sqrt_3Sqrttraining_4/Adam/clip_by_value_3*
T0*
_output_shapes

:

\
training_4/Adam/add_8/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
v
training_4/Adam/add_8Addtraining_4/Adam/Sqrt_3training_4/Adam/add_8/y*
T0*
_output_shapes

:

|
training_4/Adam/truediv_3RealDivtraining_4/Adam/mul_15training_4/Adam/add_8*
_output_shapes

:
*
T0
p
!training_4/Adam/ReadVariableOp_22ReadVariableOpdense_5/kernel*
dtype0*
_output_shapes

:

Д
training_4/Adam/sub_10Sub!training_4/Adam/ReadVariableOp_22training_4/Adam/truediv_3*
T0*
_output_shapes

:

v
"training_4/Adam/AssignVariableOp_6AssignVariableOptraining_4/Adam/Variable_2training_4/Adam/add_6*
dtype0
б
!training_4/Adam/ReadVariableOp_23ReadVariableOptraining_4/Adam/Variable_2#^training_4/Adam/AssignVariableOp_6*
dtype0*
_output_shapes

:

v
"training_4/Adam/AssignVariableOp_7AssignVariableOptraining_4/Adam/Variable_6training_4/Adam/add_7*
dtype0
б
!training_4/Adam/ReadVariableOp_24ReadVariableOptraining_4/Adam/Variable_6#^training_4/Adam/AssignVariableOp_7*
_output_shapes

:
*
dtype0
k
"training_4/Adam/AssignVariableOp_8AssignVariableOpdense_5/kerneltraining_4/Adam/sub_10*
dtype0
Х
!training_4/Adam/ReadVariableOp_25ReadVariableOpdense_5/kernel#^training_4/Adam/AssignVariableOp_8*
dtype0*
_output_shapes

:

g
!training_4/Adam/ReadVariableOp_26ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
|
%training_4/Adam/mul_16/ReadVariableOpReadVariableOptraining_4/Adam/Variable_3*
dtype0*
_output_shapes
:

М
training_4/Adam/mul_16Mul!training_4/Adam/ReadVariableOp_26%training_4/Adam/mul_16/ReadVariableOp*
_output_shapes
:
*
T0
g
!training_4/Adam/ReadVariableOp_27ReadVariableOpAdam_1/beta_1*
dtype0*
_output_shapes
: 
]
training_4/Adam/sub_11/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
{
training_4/Adam/sub_11Subtraining_4/Adam/sub_11/x!training_4/Adam/ReadVariableOp_27*
T0*
_output_shapes
: 
Р
training_4/Adam/mul_17Multraining_4/Adam/sub_114training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes
:
*
T0
q
training_4/Adam/add_9Addtraining_4/Adam/mul_16training_4/Adam/mul_17*
_output_shapes
:
*
T0
g
!training_4/Adam/ReadVariableOp_28ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
|
%training_4/Adam/mul_18/ReadVariableOpReadVariableOptraining_4/Adam/Variable_7*
dtype0*
_output_shapes
:

М
training_4/Adam/mul_18Mul!training_4/Adam/ReadVariableOp_28%training_4/Adam/mul_18/ReadVariableOp*
_output_shapes
:
*
T0
g
!training_4/Adam/ReadVariableOp_29ReadVariableOpAdam_1/beta_2*
dtype0*
_output_shapes
: 
]
training_4/Adam/sub_12/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
{
training_4/Adam/sub_12Subtraining_4/Adam/sub_12/x!training_4/Adam/ReadVariableOp_29*
T0*
_output_shapes
: 
}
training_4/Adam/Square_3Square4training_4/Adam/gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes
:
*
T0
t
training_4/Adam/mul_19Multraining_4/Adam/sub_12training_4/Adam/Square_3*
_output_shapes
:
*
T0
r
training_4/Adam/add_10Addtraining_4/Adam/mul_18training_4/Adam/mul_19*
_output_shapes
:
*
T0
n
training_4/Adam/mul_20Multraining_4/Adam/multraining_4/Adam/add_9*
_output_shapes
:
*
T0
\
training_4/Adam/Const_9Const*
dtype0*
_output_shapes
: *
valueB
 *    
]
training_4/Adam/Const_10Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Й
'training_4/Adam/clip_by_value_4/MinimumMinimumtraining_4/Adam/add_10training_4/Adam/Const_10*
_output_shapes
:
*
T0
С
training_4/Adam/clip_by_value_4Maximum'training_4/Adam/clip_by_value_4/Minimumtraining_4/Adam/Const_9*
_output_shapes
:
*
T0
d
training_4/Adam/Sqrt_4Sqrttraining_4/Adam/clip_by_value_4*
T0*
_output_shapes
:

]
training_4/Adam/add_11/yConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
t
training_4/Adam/add_11Addtraining_4/Adam/Sqrt_4training_4/Adam/add_11/y*
T0*
_output_shapes
:

y
training_4/Adam/truediv_4RealDivtraining_4/Adam/mul_20training_4/Adam/add_11*
_output_shapes
:
*
T0
j
!training_4/Adam/ReadVariableOp_30ReadVariableOpdense_5/bias*
dtype0*
_output_shapes
:

А
training_4/Adam/sub_13Sub!training_4/Adam/ReadVariableOp_30training_4/Adam/truediv_4*
_output_shapes
:
*
T0
v
"training_4/Adam/AssignVariableOp_9AssignVariableOptraining_4/Adam/Variable_3training_4/Adam/add_9*
dtype0
Э
!training_4/Adam/ReadVariableOp_31ReadVariableOptraining_4/Adam/Variable_3#^training_4/Adam/AssignVariableOp_9*
dtype0*
_output_shapes
:

x
#training_4/Adam/AssignVariableOp_10AssignVariableOptraining_4/Adam/Variable_7training_4/Adam/add_10*
dtype0
Ю
!training_4/Adam/ReadVariableOp_32ReadVariableOptraining_4/Adam/Variable_7$^training_4/Adam/AssignVariableOp_10*
dtype0*
_output_shapes
:

j
#training_4/Adam/AssignVariableOp_11AssignVariableOpdense_5/biastraining_4/Adam/sub_13*
dtype0
Р
!training_4/Adam/ReadVariableOp_33ReadVariableOpdense_5/bias$^training_4/Adam/AssignVariableOp_11*
dtype0*
_output_shapes
:

Є
training_5/group_depsNoOp^loss_2/mul^metrics_2/acc/div_no_nan"^training_4/Adam/ReadVariableOp_15"^training_4/Adam/ReadVariableOp_16"^training_4/Adam/ReadVariableOp_17"^training_4/Adam/ReadVariableOp_23"^training_4/Adam/ReadVariableOp_24"^training_4/Adam/ReadVariableOp_25"^training_4/Adam/ReadVariableOp_31"^training_4/Adam/ReadVariableOp_32"^training_4/Adam/ReadVariableOp_33!^training_4/Adam/ReadVariableOp_7!^training_4/Adam/ReadVariableOp_8!^training_4/Adam/ReadVariableOp_9
N
VarIsInitializedOp_37VarIsInitializedOp	Adam_1/lr*
_output_shapes
: 
V
VarIsInitializedOp_38VarIsInitializedOpAdam_1/iterations*
_output_shapes
: 
_
VarIsInitializedOp_39VarIsInitializedOptraining_4/Adam/Variable_9*
_output_shapes
: 
S
VarIsInitializedOp_40VarIsInitializedOpdense_5/kernel*
_output_shapes
: 
S
VarIsInitializedOp_41VarIsInitializedOpdense_4/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_42VarIsInitializedOpdense_4/bias*
_output_shapes
: 
`
VarIsInitializedOp_43VarIsInitializedOptraining_4/Adam/Variable_11*
_output_shapes
: 
_
VarIsInitializedOp_44VarIsInitializedOptraining_4/Adam/Variable_1*
_output_shapes
: 
L
VarIsInitializedOp_45VarIsInitializedOptotal_2*
_output_shapes
: 
R
VarIsInitializedOp_46VarIsInitializedOpAdam_1/beta_1*
_output_shapes
: 
_
VarIsInitializedOp_47VarIsInitializedOptraining_4/Adam/Variable_5*
_output_shapes
: 
`
VarIsInitializedOp_48VarIsInitializedOptraining_4/Adam/Variable_10*
_output_shapes
: 
Q
VarIsInitializedOp_49VarIsInitializedOpdense_5/bias*
_output_shapes
: 
]
VarIsInitializedOp_50VarIsInitializedOptraining_4/Adam/Variable*
_output_shapes
: 
_
VarIsInitializedOp_51VarIsInitializedOptraining_4/Adam/Variable_6*
_output_shapes
: 
_
VarIsInitializedOp_52VarIsInitializedOptraining_4/Adam/Variable_4*
_output_shapes
: 
Q
VarIsInitializedOp_53VarIsInitializedOpAdam_1/decay*
_output_shapes
: 
_
VarIsInitializedOp_54VarIsInitializedOptraining_4/Adam/Variable_7*
_output_shapes
: 
R
VarIsInitializedOp_55VarIsInitializedOpAdam_1/beta_2*
_output_shapes
: 
_
VarIsInitializedOp_56VarIsInitializedOptraining_4/Adam/Variable_8*
_output_shapes
: 
_
VarIsInitializedOp_57VarIsInitializedOptraining_4/Adam/Variable_2*
_output_shapes
: 
_
VarIsInitializedOp_58VarIsInitializedOptraining_4/Adam/Variable_3*
_output_shapes
: 
L
VarIsInitializedOp_59VarIsInitializedOpcount_2*
_output_shapes
: 
о
init_2NoOp^Adam_1/beta_1/Assign^Adam_1/beta_2/Assign^Adam_1/decay/Assign^Adam_1/iterations/Assign^Adam_1/lr/Assign^count_2/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^dense_5/bias/Assign^dense_5/kernel/Assign^total_2/Assign ^training_4/Adam/Variable/Assign"^training_4/Adam/Variable_1/Assign#^training_4/Adam/Variable_10/Assign#^training_4/Adam/Variable_11/Assign"^training_4/Adam/Variable_2/Assign"^training_4/Adam/Variable_3/Assign"^training_4/Adam/Variable_4/Assign"^training_4/Adam/Variable_5/Assign"^training_4/Adam/Variable_6/Assign"^training_4/Adam/Variable_7/Assign"^training_4/Adam/Variable_8/Assign"^training_4/Adam/Variable_9/Assign""нИ
cond_contextЫИЧИ
╚
cond/cond_textcond/pred_id:0cond/switch_t:0 *Т
Relu:0
cond/dropout/Floor:0
cond/dropout/Shape/Switch:1
cond/dropout/Shape:0
cond/dropout/add:0
cond/dropout/mul:0
+cond/dropout/random_uniform/RandomUniform:0
!cond/dropout/random_uniform/max:0
!cond/dropout/random_uniform/min:0
!cond/dropout/random_uniform/mul:0
!cond/dropout/random_uniform/sub:0
cond/dropout/random_uniform:0
cond/dropout/rate:0
cond/dropout/sub/x:0
cond/dropout/sub:0
cond/dropout/truediv:0
cond/pred_id:0
cond/switch_t:0%
Relu:0cond/dropout/Shape/Switch:1 
cond/pred_id:0cond/pred_id:0
╠
cond/cond_text_1cond/pred_id:0cond/switch_f:0*Ц
Relu:0
cond/Identity/Switch:0
cond/Identity:0
cond/pred_id:0
cond/switch_f:0 
Relu:0cond/Identity/Switch:0 
cond/pred_id:0cond/pred_id:0
ц
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textRloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *ф
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0Ь
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1и
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
чY
Tloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*И*
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
jloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
kloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Лloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Бloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
|loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
wloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Sloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0╘
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0├
Gloss/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0xloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0и
Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Rloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0╞
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0zloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0╫
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:02Т#
П#
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textlloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *┐ 
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Оloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Дloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Аloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Лloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Вloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Бloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
|loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Жloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Еloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
loss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
wloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
uloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0▄
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0┘
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Лloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Ш
Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0╓
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Йloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Ф
Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Зloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:02┼

┬

nloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Є
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0т
qloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0mloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0▄
lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0lloss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
Э
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_textOloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *д
Zloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0в
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
╒
Qloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*▄
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
\loss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0д
Iloss/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0в
Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Oloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0г
Hloss/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0а
Eloss/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Wloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0й
Ploss/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0Uloss/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
·
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *╛
Relu_1:0
cond_1/dropout/Floor:0
cond_1/dropout/Shape/Switch:1
cond_1/dropout/Shape:0
cond_1/dropout/add:0
cond_1/dropout/mul:0
-cond_1/dropout/random_uniform/RandomUniform:0
#cond_1/dropout/random_uniform/max:0
#cond_1/dropout/random_uniform/min:0
#cond_1/dropout/random_uniform/mul:0
#cond_1/dropout/random_uniform/sub:0
cond_1/dropout/random_uniform:0
cond_1/dropout/rate:0
cond_1/dropout/sub/x:0
cond_1/dropout/sub:0
cond_1/dropout/truediv:0
cond_1/pred_id:0
cond_1/switch_t:0$
cond_1/pred_id:0cond_1/pred_id:0)
Relu_1:0cond_1/dropout/Shape/Switch:1
ф
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*и
Relu_1:0
cond_1/Identity/Switch:0
cond_1/Identity:0
cond_1/pred_id:0
cond_1/switch_f:0$
cond_1/pred_id:0cond_1/pred_id:0$
Relu_1:0cond_1/Identity/Switch:0
■
Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textTloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *Ў
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0а
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1м
Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
н[
Vloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*ч*
lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
lloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
mloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Вloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Нloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Иloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Гloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
~loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Иloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Бloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Uloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Iloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0╩
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0|loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0╪
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0╟
Iloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0zloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0█
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0м
Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Tloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:02█#
╪#
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textnloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *В!
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Рloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Жloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Вloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Нloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Иloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Дloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Гloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
~loss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Иloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Зloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Бloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0┌
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Ь
Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Лloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Ш
Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Йloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0р
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0▌
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Нloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:12▌

┌

ploss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Д
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0р
nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0nloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ц
sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0oloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
н
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_textQloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *о
\loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0ж
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
С
Sloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*Т
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
^loss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0д
Gloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0н
Rloss_1/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0Wloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0и
Kloss_1/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0ж
Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Qloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0з
Jloss_1/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Yloss_1/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
·
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *╛
Relu_2:0
cond_2/dropout/Floor:0
cond_2/dropout/Shape/Switch:1
cond_2/dropout/Shape:0
cond_2/dropout/add:0
cond_2/dropout/mul:0
-cond_2/dropout/random_uniform/RandomUniform:0
#cond_2/dropout/random_uniform/max:0
#cond_2/dropout/random_uniform/min:0
#cond_2/dropout/random_uniform/mul:0
#cond_2/dropout/random_uniform/sub:0
cond_2/dropout/random_uniform:0
cond_2/dropout/rate:0
cond_2/dropout/sub/x:0
cond_2/dropout/sub:0
cond_2/dropout/truediv:0
cond_2/pred_id:0
cond_2/switch_t:0$
cond_2/pred_id:0cond_2/pred_id:0)
Relu_2:0cond_2/dropout/Shape/Switch:1
ф
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*и
Relu_2:0
cond_2/Identity/Switch:0
cond_2/Identity:0
cond_2/pred_id:0
cond_2/switch_f:0$
Relu_2:0cond_2/Identity/Switch:0$
cond_2/pred_id:0cond_2/pred_id:0
■
Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_textTloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *Ў
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0а
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1м
Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
н[
Vloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*ч*
lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
lloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
mloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Вloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Нloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Иloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Гloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
~loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Иloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Бloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Uloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Iloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0м
Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0Tloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0╩
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/rank:0|loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0█
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0╪
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0╟
Iloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/rank:0zloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:02█#
╪#
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textnloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *В!
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Рloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Жloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Вloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Нloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Иloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Дloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Гloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
~loss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Иloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Зloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Бloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0▌
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Нloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1┌
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Ш
Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Йloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0Ь
Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Лloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0р
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02▌

┌

ploss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Д
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ц
sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0oloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0р
nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0nloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
н
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_textQloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *о
\loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0ж
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
С
Sloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0*Т
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
^loss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0ж
Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0Qloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0и
Kloss_2/output_1_loss/broadcast_weights/assert_broadcastable/weights/shape:0Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0з
Jloss_2/output_1_loss/broadcast_weights/assert_broadcastable/values/shape:0Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0н
Rloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0Wloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0д
Gloss_2/output_1_loss/broadcast_weights/assert_broadcastable/is_scalar:0Yloss_2/output_1_loss/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0"Т8
trainable_variables·7ў7
Г
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
В
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
К
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
К
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
К
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
К
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
К
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
К
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
К
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
К
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
К
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
О
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
О
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08

SGD/iterations:0SGD/iterations/Assign$SGD/iterations/Read/ReadVariableOp:0(2*SGD/iterations/Initializer/initial_value:08
_
SGD/lr:0SGD/lr/AssignSGD/lr/Read/ReadVariableOp:0(2"SGD/lr/Initializer/initial_value:08
w
SGD/momentum:0SGD/momentum/Assign"SGD/momentum/Read/ReadVariableOp:0(2(SGD/momentum/Initializer/initial_value:08
k
SGD/decay:0SGD/decay/AssignSGD/decay/Read/ReadVariableOp:0(2%SGD/decay/Initializer/initial_value:08
А
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
А
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
Ж
training_2/SGD/Variable:0training_2/SGD/Variable/Assign-training_2/SGD/Variable/Read/ReadVariableOp:0(2training_2/SGD/zeros:08
О
training_2/SGD/Variable_1:0 training_2/SGD/Variable_1/Assign/training_2/SGD/Variable_1/Read/ReadVariableOp:0(2training_2/SGD/zeros_1:08
О
training_2/SGD/Variable_2:0 training_2/SGD/Variable_2/Assign/training_2/SGD/Variable_2/Read/ReadVariableOp:0(2training_2/SGD/zeros_2:08
О
training_2/SGD/Variable_3:0 training_2/SGD/Variable_3/Assign/training_2/SGD/Variable_3/Read/ReadVariableOp:0(2training_2/SGD/zeros_3:08
Л
Adam_1/iterations:0Adam_1/iterations/Assign'Adam_1/iterations/Read/ReadVariableOp:0(2-Adam_1/iterations/Initializer/initial_value:08
k
Adam_1/lr:0Adam_1/lr/AssignAdam_1/lr/Read/ReadVariableOp:0(2%Adam_1/lr/Initializer/initial_value:08
{
Adam_1/beta_1:0Adam_1/beta_1/Assign#Adam_1/beta_1/Read/ReadVariableOp:0(2)Adam_1/beta_1/Initializer/initial_value:08
{
Adam_1/beta_2:0Adam_1/beta_2/Assign#Adam_1/beta_2/Read/ReadVariableOp:0(2)Adam_1/beta_2/Initializer/initial_value:08
w
Adam_1/decay:0Adam_1/decay/Assign"Adam_1/decay/Read/ReadVariableOp:0(2(Adam_1/decay/Initializer/initial_value:08
А
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08
А
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08
К
training_4/Adam/Variable:0training_4/Adam/Variable/Assign.training_4/Adam/Variable/Read/ReadVariableOp:0(2training_4/Adam/zeros:08
Т
training_4/Adam/Variable_1:0!training_4/Adam/Variable_1/Assign0training_4/Adam/Variable_1/Read/ReadVariableOp:0(2training_4/Adam/zeros_1:08
Т
training_4/Adam/Variable_2:0!training_4/Adam/Variable_2/Assign0training_4/Adam/Variable_2/Read/ReadVariableOp:0(2training_4/Adam/zeros_2:08
Т
training_4/Adam/Variable_3:0!training_4/Adam/Variable_3/Assign0training_4/Adam/Variable_3/Read/ReadVariableOp:0(2training_4/Adam/zeros_3:08
Т
training_4/Adam/Variable_4:0!training_4/Adam/Variable_4/Assign0training_4/Adam/Variable_4/Read/ReadVariableOp:0(2training_4/Adam/zeros_4:08
Т
training_4/Adam/Variable_5:0!training_4/Adam/Variable_5/Assign0training_4/Adam/Variable_5/Read/ReadVariableOp:0(2training_4/Adam/zeros_5:08
Т
training_4/Adam/Variable_6:0!training_4/Adam/Variable_6/Assign0training_4/Adam/Variable_6/Read/ReadVariableOp:0(2training_4/Adam/zeros_6:08
Т
training_4/Adam/Variable_7:0!training_4/Adam/Variable_7/Assign0training_4/Adam/Variable_7/Read/ReadVariableOp:0(2training_4/Adam/zeros_7:08
Т
training_4/Adam/Variable_8:0!training_4/Adam/Variable_8/Assign0training_4/Adam/Variable_8/Read/ReadVariableOp:0(2training_4/Adam/zeros_8:08
Т
training_4/Adam/Variable_9:0!training_4/Adam/Variable_9/Assign0training_4/Adam/Variable_9/Read/ReadVariableOp:0(2training_4/Adam/zeros_9:08
Ц
training_4/Adam/Variable_10:0"training_4/Adam/Variable_10/Assign1training_4/Adam/Variable_10/Read/ReadVariableOp:0(2training_4/Adam/zeros_10:08
Ц
training_4/Adam/Variable_11:0"training_4/Adam/Variable_11/Assign1training_4/Adam/Variable_11/Read/ReadVariableOp:0(2training_4/Adam/zeros_11:08"И8
	variables·7ў7
Г
Adam/iterations:0Adam/iterations/Assign%Adam/iterations/Read/ReadVariableOp:0(2+Adam/iterations/Initializer/initial_value:08
c
	Adam/lr:0Adam/lr/AssignAdam/lr/Read/ReadVariableOp:0(2#Adam/lr/Initializer/initial_value:08
s
Adam/beta_1:0Adam/beta_1/Assign!Adam/beta_1/Read/ReadVariableOp:0(2'Adam/beta_1/Initializer/initial_value:08
s
Adam/beta_2:0Adam/beta_2/Assign!Adam/beta_2/Read/ReadVariableOp:0(2'Adam/beta_2/Initializer/initial_value:08
o
Adam/decay:0Adam/decay/Assign Adam/decay/Read/ReadVariableOp:0(2&Adam/decay/Initializer/initial_value:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
В
training/Adam/Variable:0training/Adam/Variable/Assign,training/Adam/Variable/Read/ReadVariableOp:0(2training/Adam/zeros:08
К
training/Adam/Variable_1:0training/Adam/Variable_1/Assign.training/Adam/Variable_1/Read/ReadVariableOp:0(2training/Adam/zeros_1:08
К
training/Adam/Variable_2:0training/Adam/Variable_2/Assign.training/Adam/Variable_2/Read/ReadVariableOp:0(2training/Adam/zeros_2:08
К
training/Adam/Variable_3:0training/Adam/Variable_3/Assign.training/Adam/Variable_3/Read/ReadVariableOp:0(2training/Adam/zeros_3:08
К
training/Adam/Variable_4:0training/Adam/Variable_4/Assign.training/Adam/Variable_4/Read/ReadVariableOp:0(2training/Adam/zeros_4:08
К
training/Adam/Variable_5:0training/Adam/Variable_5/Assign.training/Adam/Variable_5/Read/ReadVariableOp:0(2training/Adam/zeros_5:08
К
training/Adam/Variable_6:0training/Adam/Variable_6/Assign.training/Adam/Variable_6/Read/ReadVariableOp:0(2training/Adam/zeros_6:08
К
training/Adam/Variable_7:0training/Adam/Variable_7/Assign.training/Adam/Variable_7/Read/ReadVariableOp:0(2training/Adam/zeros_7:08
К
training/Adam/Variable_8:0training/Adam/Variable_8/Assign.training/Adam/Variable_8/Read/ReadVariableOp:0(2training/Adam/zeros_8:08
К
training/Adam/Variable_9:0training/Adam/Variable_9/Assign.training/Adam/Variable_9/Read/ReadVariableOp:0(2training/Adam/zeros_9:08
О
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign/training/Adam/Variable_10/Read/ReadVariableOp:0(2training/Adam/zeros_10:08
О
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign/training/Adam/Variable_11/Read/ReadVariableOp:0(2training/Adam/zeros_11:08

SGD/iterations:0SGD/iterations/Assign$SGD/iterations/Read/ReadVariableOp:0(2*SGD/iterations/Initializer/initial_value:08
_
SGD/lr:0SGD/lr/AssignSGD/lr/Read/ReadVariableOp:0(2"SGD/lr/Initializer/initial_value:08
w
SGD/momentum:0SGD/momentum/Assign"SGD/momentum/Read/ReadVariableOp:0(2(SGD/momentum/Initializer/initial_value:08
k
SGD/decay:0SGD/decay/AssignSGD/decay/Read/ReadVariableOp:0(2%SGD/decay/Initializer/initial_value:08
А
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
А
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
Ж
training_2/SGD/Variable:0training_2/SGD/Variable/Assign-training_2/SGD/Variable/Read/ReadVariableOp:0(2training_2/SGD/zeros:08
О
training_2/SGD/Variable_1:0 training_2/SGD/Variable_1/Assign/training_2/SGD/Variable_1/Read/ReadVariableOp:0(2training_2/SGD/zeros_1:08
О
training_2/SGD/Variable_2:0 training_2/SGD/Variable_2/Assign/training_2/SGD/Variable_2/Read/ReadVariableOp:0(2training_2/SGD/zeros_2:08
О
training_2/SGD/Variable_3:0 training_2/SGD/Variable_3/Assign/training_2/SGD/Variable_3/Read/ReadVariableOp:0(2training_2/SGD/zeros_3:08
Л
Adam_1/iterations:0Adam_1/iterations/Assign'Adam_1/iterations/Read/ReadVariableOp:0(2-Adam_1/iterations/Initializer/initial_value:08
k
Adam_1/lr:0Adam_1/lr/AssignAdam_1/lr/Read/ReadVariableOp:0(2%Adam_1/lr/Initializer/initial_value:08
{
Adam_1/beta_1:0Adam_1/beta_1/Assign#Adam_1/beta_1/Read/ReadVariableOp:0(2)Adam_1/beta_1/Initializer/initial_value:08
{
Adam_1/beta_2:0Adam_1/beta_2/Assign#Adam_1/beta_2/Read/ReadVariableOp:0(2)Adam_1/beta_2/Initializer/initial_value:08
w
Adam_1/decay:0Adam_1/decay/Assign"Adam_1/decay/Read/ReadVariableOp:0(2(Adam_1/decay/Initializer/initial_value:08
А
dense_4/kernel:0dense_4/kernel/Assign$dense_4/kernel/Read/ReadVariableOp:0(2+dense_4/kernel/Initializer/random_uniform:08
o
dense_4/bias:0dense_4/bias/Assign"dense_4/bias/Read/ReadVariableOp:0(2 dense_4/bias/Initializer/zeros:08
А
dense_5/kernel:0dense_5/kernel/Assign$dense_5/kernel/Read/ReadVariableOp:0(2+dense_5/kernel/Initializer/random_uniform:08
o
dense_5/bias:0dense_5/bias/Assign"dense_5/bias/Read/ReadVariableOp:0(2 dense_5/bias/Initializer/zeros:08
К
training_4/Adam/Variable:0training_4/Adam/Variable/Assign.training_4/Adam/Variable/Read/ReadVariableOp:0(2training_4/Adam/zeros:08
Т
training_4/Adam/Variable_1:0!training_4/Adam/Variable_1/Assign0training_4/Adam/Variable_1/Read/ReadVariableOp:0(2training_4/Adam/zeros_1:08
Т
training_4/Adam/Variable_2:0!training_4/Adam/Variable_2/Assign0training_4/Adam/Variable_2/Read/ReadVariableOp:0(2training_4/Adam/zeros_2:08
Т
training_4/Adam/Variable_3:0!training_4/Adam/Variable_3/Assign0training_4/Adam/Variable_3/Read/ReadVariableOp:0(2training_4/Adam/zeros_3:08
Т
training_4/Adam/Variable_4:0!training_4/Adam/Variable_4/Assign0training_4/Adam/Variable_4/Read/ReadVariableOp:0(2training_4/Adam/zeros_4:08
Т
training_4/Adam/Variable_5:0!training_4/Adam/Variable_5/Assign0training_4/Adam/Variable_5/Read/ReadVariableOp:0(2training_4/Adam/zeros_5:08
Т
training_4/Adam/Variable_6:0!training_4/Adam/Variable_6/Assign0training_4/Adam/Variable_6/Read/ReadVariableOp:0(2training_4/Adam/zeros_6:08
Т
training_4/Adam/Variable_7:0!training_4/Adam/Variable_7/Assign0training_4/Adam/Variable_7/Read/ReadVariableOp:0(2training_4/Adam/zeros_7:08
Т
training_4/Adam/Variable_8:0!training_4/Adam/Variable_8/Assign0training_4/Adam/Variable_8/Read/ReadVariableOp:0(2training_4/Adam/zeros_8:08
Т
training_4/Adam/Variable_9:0!training_4/Adam/Variable_9/Assign0training_4/Adam/Variable_9/Read/ReadVariableOp:0(2training_4/Adam/zeros_9:08
Ц
training_4/Adam/Variable_10:0"training_4/Adam/Variable_10/Assign1training_4/Adam/Variable_10/Read/ReadVariableOp:0(2training_4/Adam/zeros_10:08
Ц
training_4/Adam/Variable_11:0"training_4/Adam/Variable_11/Assign1training_4/Adam/Variable_11/Read/ReadVariableOp:0(2training_4/Adam/zeros_11:08[▀E       ┘▄2	╖∙г╫A*


epoch_loss8▄$?0ьдс       `/▀#	H├∙г╫A*

	epoch_accu'M?ano8        )эйP	qжи╫A*


epoch_lossAи╘>■╫i       QKD	└ жи╫A*

	epoch_accзy_?зaMЮ        )эйP	╡йн╫A*


epoch_lossG└>eА╔╣       QKD	гмн╫A*

	epoch_acc╗Lb?oЯЫ        )эйP	Т╦В▒╫A*


epoch_lossЪ ╡>	∙iв       QKD	│═В▒╫A*

	epoch_acc╠дc?v╪зь        )эйP	Шj╢╫A*


epoch_loss╢п>|_▒D       QKD	°l╢╫A*

	epoch_acc─d?Sbеы