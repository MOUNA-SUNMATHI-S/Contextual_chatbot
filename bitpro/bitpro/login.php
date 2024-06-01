<?php

// Create connection
$conn = new mysqli ($host, $dbusername, $dbpassword, $dbname);

if (mysqli_connect_error()){
  die('Connect Error ('. mysqli_connect_errno() .') '
    . mysqli_connect_error());
}
else{
  $SELECT = "SELECT username From login Where username = ? Limit 1";
  $INSERT = "INSERT Into login (username, password)values(?,?)";
}
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST["username"];
    $password = $_POST["password"];
    
    if (password_verify($password_from_database,$password)) 
    {
    // Passwords match; redirect to the welcome page or another appropriate location.
    header("Location: http://127.0.0.1:5000");
    exit();
    } else {
      // Passwords do not match; display an error message or handle as needed.
      echo "Login failed. Invalid username or password.";
    }
}
    ?>