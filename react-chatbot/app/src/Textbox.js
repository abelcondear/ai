import {chat} from './Chat.js';

let ch = new chat();

function sendMessage() {
    //const userName = "Me";
    const userName = "Marcelo Lopez";

    let ob = document.getElementById("txtInput");   
    let userInput = ob.value;   
    let userOutput = "";
    
    ob.style.backgroundColor = "gray";
    ob.readOnly = true;
    ob.placeholder="Wait a moment, please.";
    ob.value = "";
    ch.send(userInput);

    let ob_header = document.getElementsByClassName("App-header")[0];

    let child = document.createElement("label");
    child.className = "label-user"
    child.innerHTML = `<strong>${userName}</strong>: ${userInput}`;
    ob_header.appendChild(child);

    child = document.createElement("label");
    child.className = "label-ai"
    child.innerHTML = userOutput;
    ob_header.appendChild(child);

    document.ob_this.moveDownScrollBar();
}

function keyPress(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
}

function TextBox() {   
    return (
        <div class="input-area">
            <input type="text" id="txtInput" size="20" onKeyDown={(e)=>{keyPress(e)}}/>           
        </div>
        
    );
}

export default TextBox;