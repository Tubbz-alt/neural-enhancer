$( document ).ready(function() {

  function getData() {
    var input_image = document.getElementById("inputImage");
    var output_image = document.getElementById("outputImage");
       // context.getImageData(0, 0, canvas.width, canvas.height);
    $.post( "/postmethod", {
      canvas_data: JSON.stringify(input_image.src)
    }, function(err, req, resp){
    	console.log("HIRE")
      	output_image.src = 'data:image/png;base64,' + resp["responseJSON"]["data"]
    });
  }

  $( "#sendButton" ).click(function(){
    getData();
  });
});