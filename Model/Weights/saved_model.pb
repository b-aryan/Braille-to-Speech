ô
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8½
ª
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_3/depthwise_kernel
£
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*&
_output_shapes
:*
dtype0
ª
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d_3/pointwise_kernel
£
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*&
_output_shapes
:@*
dtype0

separable_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameseparable_conv2d_3/bias

+separable_conv2d_3/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias*
_output_shapes
:@*
dtype0
ª
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d_4/depthwise_kernel
£
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*&
_output_shapes
:@*
dtype0
«
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#separable_conv2d_4/pointwise_kernel
¤
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*'
_output_shapes
:@*
dtype0

separable_conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_4/bias

+separable_conv2d_4/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias*
_output_shapes	
:*
dtype0
«
#separable_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_5/depthwise_kernel
¤
7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/depthwise_kernel*'
_output_shapes
:*
dtype0
¬
#separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_5/pointwise_kernel
¥
7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/pointwise_kernel*(
_output_shapes
:*
dtype0

separable_conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_5/bias

+separable_conv2d_5/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias*
_output_shapes	
:*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¸
*Adam/separable_conv2d_3/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/m
±
>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/m*&
_output_shapes
:*
dtype0
¸
*Adam/separable_conv2d_3/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/m
±
>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/m*&
_output_shapes
:@*
dtype0

Adam/separable_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/separable_conv2d_3/bias/m

2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/m*
_output_shapes
:@*
dtype0
¸
*Adam/separable_conv2d_4/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_4/depthwise_kernel/m
±
>Adam/separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/depthwise_kernel/m*&
_output_shapes
:@*
dtype0
¹
*Adam/separable_conv2d_4/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_4/pointwise_kernel/m
²
>Adam/separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/pointwise_kernel/m*'
_output_shapes
:@*
dtype0

Adam/separable_conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_4/bias/m

2Adam/separable_conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_4/bias/m*
_output_shapes	
:*
dtype0
¹
*Adam/separable_conv2d_5/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_5/depthwise_kernel/m
²
>Adam/separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_5/depthwise_kernel/m*'
_output_shapes
:*
dtype0
º
*Adam/separable_conv2d_5/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_5/pointwise_kernel/m
³
>Adam/separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_5/pointwise_kernel/m*(
_output_shapes
:*
dtype0

Adam/separable_conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_5/bias/m

2Adam/separable_conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_5/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
¸
*Adam/separable_conv2d_3/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/v
±
>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/v*&
_output_shapes
:*
dtype0
¸
*Adam/separable_conv2d_3/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/v
±
>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/v*&
_output_shapes
:@*
dtype0

Adam/separable_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/separable_conv2d_3/bias/v

2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/v*
_output_shapes
:@*
dtype0
¸
*Adam/separable_conv2d_4/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_4/depthwise_kernel/v
±
>Adam/separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/depthwise_kernel/v*&
_output_shapes
:@*
dtype0
¹
*Adam/separable_conv2d_4/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/separable_conv2d_4/pointwise_kernel/v
²
>Adam/separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_4/pointwise_kernel/v*'
_output_shapes
:@*
dtype0

Adam/separable_conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_4/bias/v

2Adam/separable_conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_4/bias/v*
_output_shapes	
:*
dtype0
¹
*Adam/separable_conv2d_5/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_5/depthwise_kernel/v
²
>Adam/separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_5/depthwise_kernel/v*'
_output_shapes
:*
dtype0
º
*Adam/separable_conv2d_5/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_5/pointwise_kernel/v
³
>Adam/separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_5/pointwise_kernel/v*(
_output_shapes
:*
dtype0

Adam/separable_conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_5/bias/v

2Adam/separable_conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_5/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ûa
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶a
value¬aB©a B¢a
Ç
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api

depthwise_kernel
pointwise_kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$depthwise_kernel
%pointwise_kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api

/depthwise_kernel
0pointwise_kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
R
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
R
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
ì
\iter

]beta_1

^beta_2
	_decay
`learning_ratem·m¸m¹$mº%m»&m¼/m½0m¾1m¿:mÀ;mÁHmÂImÃVmÄWmÅvÆvÇvÈ$vÉ%vÊ&vË/vÌ0vÍ1vÎ:vÏ;vÐHvÑIvÒVvÓWvÔ
 
n
0
1
2
$3
%4
&5
/6
07
18
:9
;10
H11
I12
V13
W14
n
0
1
2
$3
%4
&5
/6
07
18
:9
;10
H11
I12
V13
W14
­
alayer_metrics

blayers
regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
	variables
trainable_variables
emetrics
 
 
 
 
­
flayer_metrics

glayers
regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
	variables
trainable_variables
jmetrics
yw
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

0
1
2
­
klayer_metrics

llayers
regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
	variables
trainable_variables
ometrics
 
 
 
­
player_metrics

qlayers
 regularization_losses
rnon_trainable_variables
slayer_regularization_losses
!	variables
"trainable_variables
tmetrics
yw
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1
&2

$0
%1
&2
­
ulayer_metrics

vlayers
'regularization_losses
wnon_trainable_variables
xlayer_regularization_losses
(	variables
)trainable_variables
ymetrics
 
 
 
­
zlayer_metrics

{layers
+regularization_losses
|non_trainable_variables
}layer_regularization_losses
,	variables
-trainable_variables
~metrics
yw
VARIABLE_VALUE#separable_conv2d_5/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_5/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01
12

/0
01
12
±
layer_metrics
layers
2regularization_losses
non_trainable_variables
 layer_regularization_losses
3	variables
4trainable_variables
metrics
 
 
 
²
layer_metrics
layers
6regularization_losses
non_trainable_variables
 layer_regularization_losses
7	variables
8trainable_variables
metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
²
layer_metrics
layers
<regularization_losses
non_trainable_variables
 layer_regularization_losses
=	variables
>trainable_variables
metrics
 
 
 
²
layer_metrics
layers
@regularization_losses
non_trainable_variables
 layer_regularization_losses
A	variables
Btrainable_variables
metrics
 
 
 
²
layer_metrics
layers
Dregularization_losses
non_trainable_variables
 layer_regularization_losses
E	variables
Ftrainable_variables
metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

H0
I1
²
layer_metrics
layers
Jregularization_losses
non_trainable_variables
 layer_regularization_losses
K	variables
Ltrainable_variables
metrics
 
 
 
²
layer_metrics
layers
Nregularization_losses
non_trainable_variables
  layer_regularization_losses
O	variables
Ptrainable_variables
¡metrics
 
 
 
²
¢layer_metrics
£layers
Rregularization_losses
¤non_trainable_variables
 ¥layer_regularization_losses
S	variables
Ttrainable_variables
¦metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
²
§layer_metrics
¨layers
Xregularization_losses
©non_trainable_variables
 ªlayer_regularization_losses
Y	variables
Ztrainable_variables
«metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 

¬0
­1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

®total

¯count
°	variables
±	keras_api
I

²total

³count
´
_fn_kwargs
µ	variables
¶	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

®0
¯1

°	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

²0
³1

µ	variables

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/m\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/m\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/m\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/m\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_5/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_5/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/v\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/v\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/depthwise_kernel/v\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_4/pointwise_kernel/v\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_5/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/separable_conv2d_5/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/separable_conv2d_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_2Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ó
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_800513
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
É
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_5/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_5/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_5/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_5/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_4/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_4/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_4/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_5/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_5/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_5/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*C
Tin<
:28	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_801104

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/bias#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1*Adam/separable_conv2d_3/depthwise_kernel/m*Adam/separable_conv2d_3/pointwise_kernel/mAdam/separable_conv2d_3/bias/m*Adam/separable_conv2d_4/depthwise_kernel/m*Adam/separable_conv2d_4/pointwise_kernel/mAdam/separable_conv2d_4/bias/m*Adam/separable_conv2d_5/depthwise_kernel/m*Adam/separable_conv2d_5/pointwise_kernel/mAdam/separable_conv2d_5/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m*Adam/separable_conv2d_3/depthwise_kernel/v*Adam/separable_conv2d_3/pointwise_kernel/vAdam/separable_conv2d_3/bias/v*Adam/separable_conv2d_4/depthwise_kernel/v*Adam/separable_conv2d_4/pointwise_kernel/vAdam/separable_conv2d_4/bias/v*Adam/separable_conv2d_5/depthwise_kernel/v*Adam/separable_conv2d_5/pointwise_kernel/vAdam/separable_conv2d_5/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_801276ô¶

È
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_800169

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
±
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_800187

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

Ê
-__inference_sequential_1_layer_call_fn_800462
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8004292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ú
}
(__inference_dense_5_layer_call_fn_800908

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_8002062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤
c
*__inference_dropout_2_layer_call_fn_800805

inputs
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_800815

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_800883

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_800164

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
}
(__inference_dense_4_layer_call_fn_800851

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_8001362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
c
G__inference_rescaling_1_layer_call_and_return_conditional_losses_800019

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xd
mulMulinputsCast/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
muli
addAddV2mul:z:0Cast_1/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E
ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_800283
input_2
separable_conv2d_3_800233
separable_conv2d_3_800235
separable_conv2d_3_800237
separable_conv2d_4_800241
separable_conv2d_4_800243
separable_conv2d_4_800245
separable_conv2d_5_800249
separable_conv2d_5_800251
separable_conv2d_5_800253
dense_3_800257
dense_3_800259
dense_4_800264
dense_4_800266
dense_5_800271
dense_5_800273
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCall¢*separable_conv2d_5/StatefulPartitionedCallæ
rescaling_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_8000192
rescaling_1/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall$rescaling_1/PartitionedCall:output:0separable_conv2d_3_800233separable_conv2d_3_800235separable_conv2d_3_800237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_7999002,
*separable_conv2d_3/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7999182!
max_pooling2d_2/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0separable_conv2d_4_800241separable_conv2d_4_800243separable_conv2d_4_800245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_7999412,
*separable_conv2d_4/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7999592!
max_pooling2d_3/PartitionedCall
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0separable_conv2d_5_800249separable_conv2d_5_800251separable_conv2d_5_800253*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_7999822,
*separable_conv2d_5/StatefulPartitionedCall¬
&global_max_pooling2d_1/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_8000012(
&global_max_pooling2d_1/PartitionedCall¹
dense_3/StatefulPartitionedCallStatefulPartitionedCall/global_max_pooling2d_1/PartitionedCall:output:0dense_3_800257dense_3_800259*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_8000612!
dense_3/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000942
dropout_2/PartitionedCallî
re_lu_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_8001122
re_lu_2/PartitionedCall©
dense_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_4_800264dense_4_800266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_8001362!
dense_4/StatefulPartitionedCallù
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001692
dropout_3/PartitionedCallí
re_lu_3/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_8001872
re_lu_3/PartitionedCall©
dense_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0dense_5_800271dense_5_800273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_8002062!
dense_5/StatefulPartitionedCall´
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_800264*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mulé
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¸
H
,__inference_rescaling_1_layer_call_fn_800764

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_8000192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H
¬
H__inference_sequential_1_layer_call_and_return_conditional_losses_800229
input_2
separable_conv2d_3_800027
separable_conv2d_3_800029
separable_conv2d_3_800031
separable_conv2d_4_800035
separable_conv2d_4_800037
separable_conv2d_4_800039
separable_conv2d_5_800043
separable_conv2d_5_800045
separable_conv2d_5_800047
dense_3_800072
dense_3_800074
dense_4_800147
dense_4_800149
dense_5_800217
dense_5_800219
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCall¢*separable_conv2d_5/StatefulPartitionedCallæ
rescaling_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_8000192
rescaling_1/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall$rescaling_1/PartitionedCall:output:0separable_conv2d_3_800027separable_conv2d_3_800029separable_conv2d_3_800031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_7999002,
*separable_conv2d_3/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7999182!
max_pooling2d_2/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0separable_conv2d_4_800035separable_conv2d_4_800037separable_conv2d_4_800039*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_7999412,
*separable_conv2d_4/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7999592!
max_pooling2d_3/PartitionedCall
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0separable_conv2d_5_800043separable_conv2d_5_800045separable_conv2d_5_800047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_7999822,
*separable_conv2d_5/StatefulPartitionedCall¬
&global_max_pooling2d_1/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_8000012(
&global_max_pooling2d_1/PartitionedCall¹
dense_3/StatefulPartitionedCallStatefulPartitionedCall/global_max_pooling2d_1/PartitionedCall:output:0dense_3_800072dense_3_800074*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_8000612!
dense_3/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000892#
!dropout_2/StatefulPartitionedCallö
re_lu_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_8001122
re_lu_2/PartitionedCall©
dense_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_4_800147dense_4_800149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_8001362!
dense_4/StatefulPartitionedCallµ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001642#
!dropout_3/StatefulPartitionedCallõ
re_lu_3/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_8001872
re_lu_3/PartitionedCall©
dense_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0dense_5_800217dense_5_800219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_8002062!
dense_5/StatefulPartitionedCall´
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_800147*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul±
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ô
«
C__inference_dense_3_layer_call_and_return_conditional_losses_800061

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
ð
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_799900

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identity³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate÷
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


É
-__inference_sequential_1_layer_call_fn_800751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8004292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_800094

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
c
G__inference_rescaling_1_layer_call_and_return_conditional_losses_800759

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xd
mulMulinputsCast/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
muli
addAddV2mul:z:0Cast_1/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_799959

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


É
-__inference_sequential_1_layer_call_fn_800716

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8003402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
a
¦
H__inference_sequential_1_layer_call_and_return_conditional_losses_800681

inputs?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/addì
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpò
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_3/separable_conv2d/Shape·
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate«
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativerescaling_1/add:z:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dÅ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpÞ
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
separable_conv2d_3/BiasAdd
separable_conv2d_3/ReluRelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
separable_conv2d_3/ReluÑ
max_pooling2d_2/MaxPoolMaxPool%separable_conv2d_3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolì
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpó
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2+
)separable_conv2d_4/separable_conv2d/Shape·
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate¸
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_2/MaxPool:output:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise®
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dÆ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpß
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_4/BiasAdd
separable_conv2d_4/ReluRelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_4/ReluÒ
max_pooling2d_3/MaxPoolMaxPool%separable_conv2d_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolí
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpô
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/Shape·
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rate¹
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_3/MaxPool:output:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwise®
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2dÆ
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOpß
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_5/BiasAdd
separable_conv2d_5/ReluRelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_5/Relu­
,global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2.
,global_max_pooling2d_1/Max/reduction_indicesÐ
global_max_pooling2d_1/MaxMax%separable_conv2d_5/Relu:activations:05global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling2d_1/Max§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp©
dense_3/MatMulMatMul#global_max_pooling2d_1/Max:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdd
dropout_2/IdentityIdentitydense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Identityt
re_lu_2/ReluReludropout_2/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_2/Relu¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulre_lu_2/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/BiasAdd
dropout_3/IdentityIdentitydense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/Identitys
re_lu_3/ReluReludropout_3/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_3/Relu¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulre_lu_3/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/SoftmaxÌ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mulm
IdentityIdentitydense_5/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_re_lu_2_layer_call_fn_800820

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_8001122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

Ê
-__inference_sequential_1_layer_call_fn_800373
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_8003402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ý

3__inference_separable_conv2d_5_layer_call_fn_799994

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_7999822
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
«
C__inference_dense_4_layer_call_and_return_conditional_losses_800842

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddÄ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/muld
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
L
0__inference_max_pooling2d_2_layer_call_fn_799924

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7999182
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_2_layer_call_fn_800810

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
«
C__inference_dense_3_layer_call_and_return_conditional_losses_800774

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
äs
¦
H__inference_sequential_1_layer_call_and_return_conditional_losses_800604

inputs?
;separable_conv2d_3_separable_conv2d_readvariableop_resourceA
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_3_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/addì
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2separable_conv2d_3/separable_conv2d/ReadVariableOpò
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype026
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_3/separable_conv2d/Shape·
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_3/separable_conv2d/dilation_rate«
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNativerescaling_1/add:z:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2/
-separable_conv2d_3/separable_conv2d/depthwise­
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2%
#separable_conv2d_3/separable_conv2dÅ
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)separable_conv2d_3/BiasAdd/ReadVariableOpÞ
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
separable_conv2d_3/BiasAdd
separable_conv2d_3/ReluRelu#separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
separable_conv2d_3/ReluÑ
max_pooling2d_2/MaxPoolMaxPool%separable_conv2d_3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolì
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype024
2separable_conv2d_4/separable_conv2d/ReadVariableOpó
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype026
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2+
)separable_conv2d_4/separable_conv2d/Shape·
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_4/separable_conv2d/dilation_rate¸
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_2/MaxPool:output:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2/
-separable_conv2d_4/separable_conv2d/depthwise®
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#separable_conv2d_4/separable_conv2dÆ
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)separable_conv2d_4/BiasAdd/ReadVariableOpß
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_4/BiasAdd
separable_conv2d_4/ReluRelu#separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_4/ReluÒ
max_pooling2d_3/MaxPoolMaxPool%separable_conv2d_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolí
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2separable_conv2d_5/separable_conv2d/ReadVariableOpô
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype026
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1¯
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2+
)separable_conv2d_5/separable_conv2d/Shape·
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      23
1separable_conv2d_5/separable_conv2d/dilation_rate¹
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_3/MaxPool:output:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2/
-separable_conv2d_5/separable_conv2d/depthwise®
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#separable_conv2d_5/separable_conv2dÆ
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)separable_conv2d_5/BiasAdd/ReadVariableOpß
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_5/BiasAdd
separable_conv2d_5/ReluRelu#separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
separable_conv2d_5/Relu­
,global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2.
,global_max_pooling2d_1/Max/reduction_indicesÐ
global_max_pooling2d_1/MaxMax%separable_conv2d_5/Relu:activations:05global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling2d_1/Max§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp©
dense_3/MatMulMatMul#global_max_pooling2d_1/Max:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const¤
dropout_2/dropout/MulMuldense_3/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mulz
dropout_2/dropout/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÓ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yç
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Cast£
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul_1t
re_lu_2/ReluReludropout_2/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_2/Relu¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulre_lu_2/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/dropout/Const£
dropout_3/dropout/MulMuldense_4/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/dropout/Mulz
dropout_3/dropout/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÒ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_3/dropout/GreaterEqual/yæ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/dropout/Cast¢
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_3/dropout/Mul_1s
re_lu_3/ReluReludropout_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_3/Relu¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulre_lu_3/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddy
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/SoftmaxÌ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mulm
IdentityIdentitydense_5/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

S
7__inference_global_max_pooling2d_1_layer_call_fn_800007

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_8000012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
óê
¥
"__inference__traced_restore_801276
file_prefix8
4assignvariableop_separable_conv2d_3_depthwise_kernel:
6assignvariableop_1_separable_conv2d_3_pointwise_kernel.
*assignvariableop_2_separable_conv2d_3_bias:
6assignvariableop_3_separable_conv2d_4_depthwise_kernel:
6assignvariableop_4_separable_conv2d_4_pointwise_kernel.
*assignvariableop_5_separable_conv2d_4_bias:
6assignvariableop_6_separable_conv2d_5_depthwise_kernel:
6assignvariableop_7_separable_conv2d_5_pointwise_kernel.
*assignvariableop_8_separable_conv2d_5_bias%
!assignvariableop_9_dense_3_kernel$
 assignvariableop_10_dense_3_bias&
"assignvariableop_11_dense_4_kernel$
 assignvariableop_12_dense_4_bias&
"assignvariableop_13_dense_5_kernel$
 assignvariableop_14_dense_5_bias!
assignvariableop_15_adam_iter#
assignvariableop_16_adam_beta_1#
assignvariableop_17_adam_beta_2"
assignvariableop_18_adam_decay*
&assignvariableop_19_adam_learning_rate
assignvariableop_20_total
assignvariableop_21_count
assignvariableop_22_total_1
assignvariableop_23_count_1B
>assignvariableop_24_adam_separable_conv2d_3_depthwise_kernel_mB
>assignvariableop_25_adam_separable_conv2d_3_pointwise_kernel_m6
2assignvariableop_26_adam_separable_conv2d_3_bias_mB
>assignvariableop_27_adam_separable_conv2d_4_depthwise_kernel_mB
>assignvariableop_28_adam_separable_conv2d_4_pointwise_kernel_m6
2assignvariableop_29_adam_separable_conv2d_4_bias_mB
>assignvariableop_30_adam_separable_conv2d_5_depthwise_kernel_mB
>assignvariableop_31_adam_separable_conv2d_5_pointwise_kernel_m6
2assignvariableop_32_adam_separable_conv2d_5_bias_m-
)assignvariableop_33_adam_dense_3_kernel_m+
'assignvariableop_34_adam_dense_3_bias_m-
)assignvariableop_35_adam_dense_4_kernel_m+
'assignvariableop_36_adam_dense_4_bias_m-
)assignvariableop_37_adam_dense_5_kernel_m+
'assignvariableop_38_adam_dense_5_bias_mB
>assignvariableop_39_adam_separable_conv2d_3_depthwise_kernel_vB
>assignvariableop_40_adam_separable_conv2d_3_pointwise_kernel_v6
2assignvariableop_41_adam_separable_conv2d_3_bias_vB
>assignvariableop_42_adam_separable_conv2d_4_depthwise_kernel_vB
>assignvariableop_43_adam_separable_conv2d_4_pointwise_kernel_v6
2assignvariableop_44_adam_separable_conv2d_4_bias_vB
>assignvariableop_45_adam_separable_conv2d_5_depthwise_kernel_vB
>assignvariableop_46_adam_separable_conv2d_5_pointwise_kernel_v6
2assignvariableop_47_adam_separable_conv2d_5_bias_v-
)assignvariableop_48_adam_dense_3_kernel_v+
'assignvariableop_49_adam_dense_3_bias_v-
)assignvariableop_50_adam_dense_4_kernel_v+
'assignvariableop_51_adam_dense_4_bias_v-
)assignvariableop_52_adam_dense_5_kernel_v+
'assignvariableop_53_adam_dense_5_bias_v
identity_55¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0* 
valueB7B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesý
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ò
_output_shapesß
Ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity³
AssignVariableOpAssignVariableOp4assignvariableop_separable_conv2d_3_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1»
AssignVariableOp_1AssignVariableOp6assignvariableop_1_separable_conv2d_3_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¯
AssignVariableOp_2AssignVariableOp*assignvariableop_2_separable_conv2d_3_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_4_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4»
AssignVariableOp_4AssignVariableOp6assignvariableop_4_separable_conv2d_4_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_separable_conv2d_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6»
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_5_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7»
AssignVariableOp_7AssignVariableOp6assignvariableop_7_separable_conv2d_5_pointwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¯
AssignVariableOp_8AssignVariableOp*assignvariableop_8_separable_conv2d_5_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_3_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¨
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_3_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_4_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¨
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_5_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¨
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_5_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15¥
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_iterIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¦
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Æ
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_separable_conv2d_3_depthwise_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Æ
AssignVariableOp_25AssignVariableOp>assignvariableop_25_adam_separable_conv2d_3_pointwise_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26º
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_separable_conv2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Æ
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_separable_conv2d_4_depthwise_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Æ
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_separable_conv2d_4_pointwise_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29º
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_separable_conv2d_4_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Æ
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_separable_conv2d_5_depthwise_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Æ
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_separable_conv2d_5_pointwise_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32º
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_separable_conv2d_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¯
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¯
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¯
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Æ
AssignVariableOp_39AssignVariableOp>assignvariableop_39_adam_separable_conv2d_3_depthwise_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Æ
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_separable_conv2d_3_pointwise_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41º
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_separable_conv2d_3_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Æ
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_separable_conv2d_4_depthwise_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Æ
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_separable_conv2d_4_pointwise_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44º
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_separable_conv2d_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Æ
AssignVariableOp_45AssignVariableOp>assignvariableop_45_adam_separable_conv2d_5_depthwise_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Æ
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_separable_conv2d_5_pointwise_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47º
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_separable_conv2d_5_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_3_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¯
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_3_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_4_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¯
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_4_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_5_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¯
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_5_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54õ	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*ï
_input_shapesÝ
Ú: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
È
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_800868

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð	
Á
$__inference_signature_wrapper_800513
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_7998832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
E
ã
H__inference_sequential_1_layer_call_and_return_conditional_losses_800429

inputs
separable_conv2d_3_800379
separable_conv2d_3_800381
separable_conv2d_3_800383
separable_conv2d_4_800387
separable_conv2d_4_800389
separable_conv2d_4_800391
separable_conv2d_5_800395
separable_conv2d_5_800397
separable_conv2d_5_800399
dense_3_800403
dense_3_800405
dense_4_800410
dense_4_800412
dense_5_800417
dense_5_800419
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCall¢*separable_conv2d_5/StatefulPartitionedCallå
rescaling_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_8000192
rescaling_1/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall$rescaling_1/PartitionedCall:output:0separable_conv2d_3_800379separable_conv2d_3_800381separable_conv2d_3_800383*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_7999002,
*separable_conv2d_3/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7999182!
max_pooling2d_2/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0separable_conv2d_4_800387separable_conv2d_4_800389separable_conv2d_4_800391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_7999412,
*separable_conv2d_4/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7999592!
max_pooling2d_3/PartitionedCall
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0separable_conv2d_5_800395separable_conv2d_5_800397separable_conv2d_5_800399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_7999822,
*separable_conv2d_5/StatefulPartitionedCall¬
&global_max_pooling2d_1/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_8000012(
&global_max_pooling2d_1/PartitionedCall¹
dense_3/StatefulPartitionedCallStatefulPartitionedCall/global_max_pooling2d_1/PartitionedCall:output:0dense_3_800403dense_3_800405*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_8000612!
dense_3/StatefulPartitionedCallú
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000942
dropout_2/PartitionedCallî
re_lu_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_8001122
re_lu_2/PartitionedCall©
dense_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_4_800410dense_4_800412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_8001362!
dense_4/StatefulPartitionedCallù
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001692
dropout_3/PartitionedCallí
re_lu_3/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_8001872
re_lu_3/PartitionedCall©
dense_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0dense_5_800417dense_5_800419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_8002062!
dense_5/StatefulPartitionedCall´
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_800410*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mulé
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

3__inference_separable_conv2d_4_layer_call_fn_799953

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_7999412
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_799918

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

3__inference_separable_conv2d_3_layer_call_fn_799912

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_7999002
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
«
C__inference_dense_5_layer_call_and_return_conditional_losses_800899

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_800800

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñj
Ã
!__inference__wrapped_model_799883
input_2L
Hsequential_1_separable_conv2d_3_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_3_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_3_biasadd_readvariableop_resourceL
Hsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_4_biasadd_readvariableop_resourceL
Hsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_5_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource7
3sequential_1_dense_4_matmul_readvariableop_resource8
4sequential_1_dense_4_biasadd_readvariableop_resource7
3sequential_1_dense_5_matmul_readvariableop_resource8
4sequential_1_dense_5_biasadd_readvariableop_resource
identity
sequential_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2!
sequential_1/rescaling_1/Cast/x
!sequential_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/rescaling_1/Cast_1/x°
sequential_1/rescaling_1/mulMulinput_2(sequential_1/rescaling_1/Cast/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/rescaling_1/mulÍ
sequential_1/rescaling_1/addAddV2 sequential_1/rescaling_1/mul:z:0*sequential_1/rescaling_1/Cast_1/x:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/rescaling_1/add
?sequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02A
?sequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOp
Asequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@*
dtype02C
Asequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1É
6sequential_1/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            28
6sequential_1/separable_conv2d_3/separable_conv2d/ShapeÑ
>sequential_1/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_1/separable_conv2d_3/separable_conv2d/dilation_rateß
:sequential_1/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative sequential_1/rescaling_1/add:z:0Gsequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2<
:sequential_1/separable_conv2d_3/separable_conv2d/depthwiseá
0sequential_1/separable_conv2d_3/separable_conv2dConv2DCsequential_1/separable_conv2d_3/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
22
0sequential_1/separable_conv2d_3/separable_conv2dì
6sequential_1/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/separable_conv2d_3/BiasAdd/ReadVariableOp
'sequential_1/separable_conv2d_3/BiasAddBiasAdd9sequential_1/separable_conv2d_3/separable_conv2d:output:0>sequential_1/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'sequential_1/separable_conv2d_3/BiasAddÀ
$sequential_1/separable_conv2d_3/ReluRelu0sequential_1/separable_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$sequential_1/separable_conv2d_3/Reluø
$sequential_1/max_pooling2d_2/MaxPoolMaxPool2sequential_1/separable_conv2d_3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_2/MaxPool
?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02A
?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp
Asequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype02C
Asequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1É
6sequential_1/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      28
6sequential_1/separable_conv2d_4/separable_conv2d/ShapeÑ
>sequential_1/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_1/separable_conv2d_4/separable_conv2d/dilation_rateì
:sequential_1/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative-sequential_1/max_pooling2d_2/MaxPool:output:0Gsequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2<
:sequential_1/separable_conv2d_4/separable_conv2d/depthwiseâ
0sequential_1/separable_conv2d_4/separable_conv2dConv2DCsequential_1/separable_conv2d_4/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
22
0sequential_1/separable_conv2d_4/separable_conv2dí
6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp
'sequential_1/separable_conv2d_4/BiasAddBiasAdd9sequential_1/separable_conv2d_4/separable_conv2d:output:0>sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_1/separable_conv2d_4/BiasAddÁ
$sequential_1/separable_conv2d_4/ReluRelu0sequential_1/separable_conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential_1/separable_conv2d_4/Reluù
$sequential_1/max_pooling2d_3/MaxPoolMaxPool2sequential_1/separable_conv2d_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_3/MaxPool
?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02A
?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp
Asequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02C
Asequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1É
6sequential_1/separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            28
6sequential_1/separable_conv2d_5/separable_conv2d/ShapeÑ
>sequential_1/separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2@
>sequential_1/separable_conv2d_5/separable_conv2d/dilation_rateí
:sequential_1/separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative-sequential_1/max_pooling2d_3/MaxPool:output:0Gsequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2<
:sequential_1/separable_conv2d_5/separable_conv2d/depthwiseâ
0sequential_1/separable_conv2d_5/separable_conv2dConv2DCsequential_1/separable_conv2d_5/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
22
0sequential_1/separable_conv2d_5/separable_conv2dí
6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp
'sequential_1/separable_conv2d_5/BiasAddBiasAdd9sequential_1/separable_conv2d_5/separable_conv2d:output:0>sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_1/separable_conv2d_5/BiasAddÁ
$sequential_1/separable_conv2d_5/ReluRelu0sequential_1/separable_conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential_1/separable_conv2d_5/ReluÇ
9sequential_1/global_max_pooling2d_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9sequential_1/global_max_pooling2d_1/Max/reduction_indices
'sequential_1/global_max_pooling2d_1/MaxMax2sequential_1/separable_conv2d_5/Relu:activations:0Bsequential_1/global_max_pooling2d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_1/global_max_pooling2d_1/MaxÎ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOpÝ
sequential_1/dense_3/MatMulMatMul0sequential_1/global_max_pooling2d_1/Max:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_3/MatMulÌ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÖ
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_3/BiasAdd¨
sequential_1/dropout_2/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_1/dropout_2/Identity
sequential_1/re_lu_2/ReluRelu(sequential_1/dropout_2/Identity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/re_lu_2/ReluÍ
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOpÓ
sequential_1/dense_4/MatMulMatMul'sequential_1/re_lu_2/Relu:activations:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_1/dense_4/MatMulË
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOpÕ
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_1/dense_4/BiasAdd§
sequential_1/dropout_3/IdentityIdentity%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_1/dropout_3/Identity
sequential_1/re_lu_3/ReluRelu(sequential_1/dropout_3/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_1/re_lu_3/ReluÌ
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOpÓ
sequential_1/dense_5/MatMulMatMul'sequential_1/re_lu_3/Relu:activations:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_5/MatMulË
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOpÕ
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_5/BiasAdd 
sequential_1/dense_5/SoftmaxSoftmax%sequential_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_5/Softmaxz
IdentityIdentity&sequential_1/dense_5/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_800863

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

D
(__inference_re_lu_3_layer_call_fn_800888

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_8001872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_800795

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_800089

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

n
__inference_loss_fn_0_800919=
9dense_4_kernel_regularizer_square_readvariableop_resource
identityß
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mule
IdentityIdentity"dense_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:

ð
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_799982

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identity´
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp»
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateø
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d/depthwiseô
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¥
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*M
_input_shapes<
::,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H
«
H__inference_sequential_1_layer_call_and_return_conditional_losses_800340

inputs
separable_conv2d_3_800290
separable_conv2d_3_800292
separable_conv2d_3_800294
separable_conv2d_4_800298
separable_conv2d_4_800300
separable_conv2d_4_800302
separable_conv2d_5_800306
separable_conv2d_5_800308
separable_conv2d_5_800310
dense_3_800314
dense_3_800316
dense_4_800321
dense_4_800323
dense_5_800328
dense_5_800330
identity¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢*separable_conv2d_3/StatefulPartitionedCall¢*separable_conv2d_4/StatefulPartitionedCall¢*separable_conv2d_5/StatefulPartitionedCallå
rescaling_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_8000192
rescaling_1/PartitionedCall
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall$rescaling_1/PartitionedCall:output:0separable_conv2d_3_800290separable_conv2d_3_800292separable_conv2d_3_800294*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_7999002,
*separable_conv2d_3/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7999182!
max_pooling2d_2/PartitionedCall
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0separable_conv2d_4_800298separable_conv2d_4_800300separable_conv2d_4_800302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_7999412,
*separable_conv2d_4/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7999592!
max_pooling2d_3/PartitionedCall
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0separable_conv2d_5_800306separable_conv2d_5_800308separable_conv2d_5_800310*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_7999822,
*separable_conv2d_5/StatefulPartitionedCall¬
&global_max_pooling2d_1/PartitionedCallPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_8000012(
&global_max_pooling2d_1/PartitionedCall¹
dense_3/StatefulPartitionedCallStatefulPartitionedCall/global_max_pooling2d_1/PartitionedCall:output:0dense_3_800314dense_3_800316*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_8000612!
dense_3/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_8000892#
!dropout_2/StatefulPartitionedCallö
re_lu_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_8001122
re_lu_2/PartitionedCall©
dense_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_4_800321dense_4_800323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_8001362!
dense_4/StatefulPartitionedCallµ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001642#
!dropout_3/StatefulPartitionedCallõ
re_lu_3/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_8001872
re_lu_3/PartitionedCall©
dense_5/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0dense_5_800328dense_5_800330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_8002062!
dense_5/StatefulPartitionedCall´
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_800321*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul±
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ:::::::::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
n
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_800001

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_800112

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
}
(__inference_dense_3_layer_call_fn_800783

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_8000612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
c
*__inference_dropout_3_layer_call_fn_800873

inputs
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í
«
C__inference_dense_4_layer_call_and_return_conditional_losses_800136

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddÄ
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp´
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	@2#
!dense_4/kernel/Regularizer/Square
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Constº
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q92"
 dense_4/kernel/Regularizer/mul/x¼
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/muld
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦w
¶
__inference__traced_save_801104
file_prefixB
>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_3_bias_read_readvariableopB
>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_4_bias_read_readvariableopB
>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_5_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_4_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_5_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_5_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_5_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_4_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_5_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_5_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_5_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1644af907cb240f388b833a2afc0cf5f/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0* 
valueB7B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names÷
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesâ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_5_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_3_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_4_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_5_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_5_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_5_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_3_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_4_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_4_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_4_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_5_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_5_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_5_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: ::@:@:@:@:::::
::	@:@:@:: : : : : : : : : ::@:@:@:@:::::
::	@:@:@:::@:@:@:@:::::
::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::-)
'
_output_shapes
::.*
(
_output_shapes
::!	

_output_shapes	
::&
"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::-)
'
_output_shapes
::. *
(
_output_shapes
::!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::%$!

_output_shapes
:	@: %

_output_shapes
:@:$& 

_output_shapes

:@: '

_output_shapes
::,((
&
_output_shapes
::,)(
&
_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@:-,)
'
_output_shapes
:@:!-

_output_shapes	
::-.)
'
_output_shapes
::./*
(
_output_shapes
::!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::%3!

_output_shapes
:	@: 4

_output_shapes
:@:$5 

_output_shapes

:@: 6

_output_shapes
::7

_output_shapes
: 
­
L
0__inference_max_pooling2d_3_layer_call_fn_799965

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7999592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
«
C__inference_dense_5_layer_call_and_return_conditional_losses_800206

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

F
*__inference_dropout_3_layer_call_fn_800878

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_8001692
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ð
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_799941

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identity³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOpº
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate÷
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
separable_conv2d/depthwiseô
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¥
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_28
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ;
dense_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:À¯
ýh
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Õ_default_save_signature
Ö__call__
+×&call_and_return_all_conditional_losses"Ùd
_tf_keras_sequentialºd{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ë
regularization_losses
	variables
trainable_variables
	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"class_name": "Rescaling", "name": "rescaling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}
é
depthwise_kernel
pointwise_kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"¢
_tf_keras_layer{"class_name": "SeparableConv2D", "name": "separable_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "separable_conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 3]}}

 regularization_losses
!	variables
"trainable_variables
#	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ì
$depthwise_kernel
%pointwise_kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "SeparableConv2D", "name": "separable_conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}

+regularization_losses
,	variables
-trainable_variables
.	keras_api
à__call__
+á&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ì
/depthwise_kernel
0pointwise_kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "SeparableConv2D", "name": "separable_conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 128]}}

6regularization_losses
7	variables
8trainable_variables
9	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"class_name": "GlobalMaxPooling2D", "name": "global_max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
÷

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ç
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
è__call__
+é&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
í
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
°

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
ì__call__
+í&call_and_return_all_conditional_losses"
_tf_keras_layerï{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ç
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
î__call__
+ï&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
í
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
õ

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ÿ
\iter

]beta_1

^beta_2
	_decay
`learning_ratem·m¸m¹$mº%m»&m¼/m½0m¾1m¿:mÀ;mÁHmÂImÃVmÄWmÅvÆvÇvÈ$vÉ%vÊ&vË/vÌ0vÍ1vÎ:vÏ;vÐHvÑIvÒVvÓWvÔ"
	optimizer
(
ô0"
trackable_list_wrapper

0
1
2
$3
%4
&5
/6
07
18
:9
;10
H11
I12
V13
W14"
trackable_list_wrapper

0
1
2
$3
%4
&5
/6
07
18
:9
;10
H11
I12
V13
W14"
trackable_list_wrapper
Î
alayer_metrics

blayers
regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
	variables
trainable_variables
emetrics
Ö__call__
Õ_default_save_signature
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
-
õserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
flayer_metrics

glayers
regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
	variables
trainable_variables
jmetrics
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
=:;2#separable_conv2d_3/depthwise_kernel
=:;@2#separable_conv2d_3/pointwise_kernel
%:#@2separable_conv2d_3/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
°
klayer_metrics

llayers
regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
	variables
trainable_variables
ometrics
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
player_metrics

qlayers
 regularization_losses
rnon_trainable_variables
slayer_regularization_losses
!	variables
"trainable_variables
tmetrics
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
=:;@2#separable_conv2d_4/depthwise_kernel
>:<@2#separable_conv2d_4/pointwise_kernel
&:$2separable_conv2d_4/bias
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
°
ulayer_metrics

vlayers
'regularization_losses
wnon_trainable_variables
xlayer_regularization_losses
(	variables
)trainable_variables
ymetrics
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
zlayer_metrics

{layers
+regularization_losses
|non_trainable_variables
}layer_regularization_losses
,	variables
-trainable_variables
~metrics
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
>:<2#separable_conv2d_5/depthwise_kernel
?:=2#separable_conv2d_5/pointwise_kernel
&:$2separable_conv2d_5/bias
 "
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
´
layer_metrics
layers
2regularization_losses
non_trainable_variables
 layer_regularization_losses
3	variables
4trainable_variables
metrics
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
layers
6regularization_losses
non_trainable_variables
 layer_regularization_losses
7	variables
8trainable_variables
metrics
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
µ
layer_metrics
layers
<regularization_losses
non_trainable_variables
 layer_regularization_losses
=	variables
>trainable_variables
metrics
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
layers
@regularization_losses
non_trainable_variables
 layer_regularization_losses
A	variables
Btrainable_variables
metrics
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
layers
Dregularization_losses
non_trainable_variables
 layer_regularization_losses
E	variables
Ftrainable_variables
metrics
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_4/kernel
:@2dense_4/bias
(
ô0"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
µ
layer_metrics
layers
Jregularization_losses
non_trainable_variables
 layer_regularization_losses
K	variables
Ltrainable_variables
metrics
ì__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
layers
Nregularization_losses
non_trainable_variables
  layer_regularization_losses
O	variables
Ptrainable_variables
¡metrics
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢layer_metrics
£layers
Rregularization_losses
¤non_trainable_variables
 ¥layer_regularization_losses
S	variables
Ttrainable_variables
¦metrics
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
µ
§layer_metrics
¨layers
Xregularization_losses
©non_trainable_variables
 ªlayer_regularization_losses
Y	variables
Ztrainable_variables
«metrics
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¬0
­1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ô0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

®total

¯count
°	variables
±	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


²total

³count
´
_fn_kwargs
µ	variables
¶	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
®0
¯1"
trackable_list_wrapper
.
°	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
²0
³1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
B:@2*Adam/separable_conv2d_3/depthwise_kernel/m
B:@@2*Adam/separable_conv2d_3/pointwise_kernel/m
*:(@2Adam/separable_conv2d_3/bias/m
B:@@2*Adam/separable_conv2d_4/depthwise_kernel/m
C:A@2*Adam/separable_conv2d_4/pointwise_kernel/m
+:)2Adam/separable_conv2d_4/bias/m
C:A2*Adam/separable_conv2d_5/depthwise_kernel/m
D:B2*Adam/separable_conv2d_5/pointwise_kernel/m
+:)2Adam/separable_conv2d_5/bias/m
':%
2Adam/dense_3/kernel/m
 :2Adam/dense_3/bias/m
&:$	@2Adam/dense_4/kernel/m
:@2Adam/dense_4/bias/m
%:#@2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
B:@2*Adam/separable_conv2d_3/depthwise_kernel/v
B:@@2*Adam/separable_conv2d_3/pointwise_kernel/v
*:(@2Adam/separable_conv2d_3/bias/v
B:@@2*Adam/separable_conv2d_4/depthwise_kernel/v
C:A@2*Adam/separable_conv2d_4/pointwise_kernel/v
+:)2Adam/separable_conv2d_4/bias/v
C:A2*Adam/separable_conv2d_5/depthwise_kernel/v
D:B2*Adam/separable_conv2d_5/pointwise_kernel/v
+:)2Adam/separable_conv2d_5/bias/v
':%
2Adam/dense_3/kernel/v
 :2Adam/dense_3/bias/v
&:$	@2Adam/dense_4/kernel/v
:@2Adam/dense_4/bias/v
%:#@2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
ç2ä
!__inference__wrapped_model_799883¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_2ÿÿÿÿÿÿÿÿÿ
2ÿ
-__inference_sequential_1_layer_call_fn_800716
-__inference_sequential_1_layer_call_fn_800462
-__inference_sequential_1_layer_call_fn_800373
-__inference_sequential_1_layer_call_fn_800751À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_800604
H__inference_sequential_1_layer_call_and_return_conditional_losses_800681
H__inference_sequential_1_layer_call_and_return_conditional_losses_800283
H__inference_sequential_1_layer_call_and_return_conditional_losses_800229À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_rescaling_1_layer_call_fn_800764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_rescaling_1_layer_call_and_return_conditional_losses_800759¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
3__inference_separable_conv2d_3_layer_call_fn_799912×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­2ª
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_799900×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_2_layer_call_fn_799924à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_799918à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
3__inference_separable_conv2d_4_layer_call_fn_799953×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
­2ª
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_799941×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
0__inference_max_pooling2d_3_layer_call_fn_799965à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_799959à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
3__inference_separable_conv2d_5_layer_call_fn_799994Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
®2«
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_799982Ø
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
7__inference_global_max_pooling2d_1_layer_call_fn_800007à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º2·
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_800001à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_dense_3_layer_call_fn_800783¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_800774¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_dropout_2_layer_call_fn_800805
*__inference_dropout_2_layer_call_fn_800810´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_2_layer_call_and_return_conditional_losses_800800
E__inference_dropout_2_layer_call_and_return_conditional_losses_800795´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_re_lu_2_layer_call_fn_800820¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_re_lu_2_layer_call_and_return_conditional_losses_800815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_4_layer_call_fn_800851¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_800842¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_dropout_3_layer_call_fn_800873
*__inference_dropout_3_layer_call_fn_800878´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_3_layer_call_and_return_conditional_losses_800868
E__inference_dropout_3_layer_call_and_return_conditional_losses_800863´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_re_lu_3_layer_call_fn_800888¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_re_lu_3_layer_call_and_return_conditional_losses_800883¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_5_layer_call_fn_800908¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_800899¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_800919
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
3B1
$__inference_signature_wrapper_800513input_2£
!__inference__wrapped_model_799883~$%&/01:;HIVW8¢5
.¢+
)&
input_2ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_3_layer_call_and_return_conditional_losses_800774^:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_3_layer_call_fn_800783Q:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_4_layer_call_and_return_conditional_losses_800842]HI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_dense_4_layer_call_fn_800851PHI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_5_layer_call_and_return_conditional_losses_800899\VW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_5_layer_call_fn_800908OVW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_2_layer_call_and_return_conditional_losses_800795^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_2_layer_call_and_return_conditional_losses_800800^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_2_layer_call_fn_800805Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_2_layer_call_fn_800810Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dropout_3_layer_call_and_return_conditional_losses_800863\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¥
E__inference_dropout_3_layer_call_and_return_conditional_losses_800868\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
*__inference_dropout_3_layer_call_fn_800873O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@}
*__inference_dropout_3_layer_call_fn_800878O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@Û
R__inference_global_max_pooling2d_1_layer_call_and_return_conditional_losses_800001R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ²
7__inference_global_max_pooling2d_1_layer_call_fn_800007wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_800919H¢

¢ 
ª " î
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_799918R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_2_layer_call_fn_799924R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_799959R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_3_layer_call_fn_799965R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
C__inference_re_lu_2_layer_call_and_return_conditional_losses_800815Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
(__inference_re_lu_2_layer_call_fn_800820M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
C__inference_re_lu_3_layer_call_and_return_conditional_losses_800883X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 w
(__inference_re_lu_3_layer_call_fn_800888K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@³
G__inference_rescaling_1_layer_call_and_return_conditional_losses_800759h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_rescaling_1_layer_call_fn_800764[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿä
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_799900I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¼
3__inference_separable_conv2d_3_layer_call_fn_799912I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@å
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_799941$%&I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ½
3__inference_separable_conv2d_4_layer_call_fn_799953$%&I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_799982/01J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
3__inference_separable_conv2d_5_layer_call_fn_799994/01J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
H__inference_sequential_1_layer_call_and_return_conditional_losses_800229z$%&/01:;HIVW@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
H__inference_sequential_1_layer_call_and_return_conditional_losses_800283z$%&/01:;HIVW@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_sequential_1_layer_call_and_return_conditional_losses_800604y$%&/01:;HIVW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_sequential_1_layer_call_and_return_conditional_losses_800681y$%&/01:;HIVW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_1_layer_call_fn_800373m$%&/01:;HIVW@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_800462m$%&/01:;HIVW@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_800716l$%&/01:;HIVW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_800751l$%&/01:;HIVW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ²
$__inference_signature_wrapper_800513$%&/01:;HIVWC¢@
¢ 
9ª6
4
input_2)&
input_2ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ