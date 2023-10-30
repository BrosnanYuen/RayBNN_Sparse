
use arrayfire;
use RayBNN_Sparse;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;




#[test]
fn test_Remove4() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);



}
