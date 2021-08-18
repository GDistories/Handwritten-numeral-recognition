let canvas_input = document.getElementById("canvas_img");
let context_input = canvas_input.getContext('2d');
let digit_output = document.getElementById("predictdigit");
let classification_visual = null;
let mouse_move = false;

// Initialize
window.onload = function(){
    
    context_input.fillStyle = "white";
    context_input.fillRect(0, 0, canvas_input.width, canvas_input.height);
    context_input.lineWidth = 7;
    context_input.lineCap = "round";
    
    init_classification();
}

// Initialize the classification map
function init_classification(){

    const origin_prob = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const layout = { top: 10, right: 10, bottom: 10, left: 20 }
        , width = 400, height = 400;

    let y_layout = d3.scaleLinear()
        .domain([9, 0])
        .range([height, 0]);
    
    classification_visual = d3.select("#classification")
        .attr("width", width + layout.left + layout.right)
        .attr("height", height + layout.top + layout.bottom)
        .append("g")
        .attr("transform", "translate(" + layout.left + "," + layout.top + ")");

    classification_visual.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(y_layout));

    const prob_bar_length = 20
    classification_visual.selectAll("svg")
        .data(origin_prob)
        .enter()
        .append("rect")
        .attr("y", function(d,i){return y_layout(i) - prob_bar_length / 2})
        .attr("height", prob_bar_length)
        .style("fill", "green")
        .attr("x", 0)
        .attr("width", function(d){return d * 2})
        .call(d3.axisLeft(y_layout));
}

// monitor mouse press
canvas_input.addEventListener("mousedown", function(e) {
    if(e.button == 0){
        let targe_rect = e.target.getBoundingClientRect();
        let x = e.clientX - targe_rect.left;
        let y = e.clientY - targe_rect.top;
        mouse_move = true;
        context_input.beginPath();
        context_input.moveTo(x, y);
    }   
    else if(e.button == 2){
        clear_funtion();  // right click for clear input
    }
});

// monitor the mouse to release
canvas_input.addEventListener("mouseup", function(e) { 
    if(e.button == 0){
        mouse_move = false; 
        recognition();
    }
});

// monitor mouse movement
canvas_input.addEventListener("mousemove", function(e) {
    let targe_rect = e.target.getBoundingClientRect();
    let x = e.clientX - targe_rect.left;
    let y = e.clientY - targe_rect.top;
    if(mouse_move){
        context_input.lineTo(x, y);
        context_input.stroke();
    }
});

// Click the clear button
document.getElementById("clear_button").onclick = clear_funtion;

function clear_funtion(){
    mouse_move = false;
    context_input.fillStyle = "white";
    context_input.fillRect(0, 0, canvas_input.width, canvas_input.height);
    context_input.fillStyle = "black";
}

// post data to server for recognition
function recognition() {
    console.time("time");

    $.ajax({
            url: './classification',
            type:'POST',
            data : {img : canvas_input.toDataURL("image/png").replace('data:image/png;base64,','') },

        }).done(function(data) {

            show_predicted(JSON.parse(data))

        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            console.log(XMLHttpRequest);
            alert("error");
        })

    console.timeEnd("time");
}


function show_predicted(result_data){

    // predicted number
    digit_output.textContent = result_data.predict_digit;

    // predict the probability of the number
    document.getElementById("probvalue").innerHTML =
        "Probability of decision : " + result_data.prob[result_data.predict_digit].toFixed(2) + "%";

    // Output probability histogram
    let graphData = [];
    for (let val in result_data.prob){
        graphData.push(result_data.prob[val]);
    }

    classification_visual.selectAll("rect")
        .data(graphData)
        .transition()
        .duration(300)
        .style("fill", function(d, i){return (i == result_data.predict_digit ? "red":"yellow")})
        .attr("width", function(d){return d * 2})
}
