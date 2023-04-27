# Instructions

## Use the web for single use

Please access the following link: https://simple-qr-code-gen.herokuapp.com/ and enter the text you want to encode in the text box. Then click the button to generate the QR code.


## Use the terminal for multiple uses

```bash
curl -X POST -H "Content-Type: application/json" -H "x-api-key: _your_api_key_here_" -d '{"text": "_your_text_here_"}' https://simple-qr-code-gen.herokuapp.com/generate_qr --output _output_name_.png
```