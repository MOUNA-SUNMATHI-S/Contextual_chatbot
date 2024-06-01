<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST'){
$funame = $_POST['funame'];
$luname = $_POST['luname'];
$email  = $_POST['email'];
$upswd= $_POST['upswd'];




if (!empty($funame) || !empty($luname)|| !empty($email) || !empty($upswd))
{

$host = "localhost";
$dbusername = "root";
$dbpassword = "";
$dbname = "bitpro";



// Create connection
$conn = new mysqli ($host, $dbusername, $dbpassword, $dbname);

if (mysqli_connect_error()){
  die('Connect Error ('. mysqli_connect_errno() .') '
    . mysqli_connect_error());
}
else{
  $SELECT = "SELECT email From register Where email = ? Limit 1";
  $INSERT = "INSERT Into register (funame, luname, email, upswd)values(?,?,?,?)";

//Prepare statement
     $stmt = $conn->prepare($SELECT);
     $stmt->bind_param("s", $email);
     $stmt->execute();
     $stmt->bind_result($email);
     $stmt->store_result();
     $rnum = $stmt->num_rows;

     //checking username
      if ($rnum==0) {
      $stmt->close();
      $stmt = $conn->prepare($INSERT);
      $stmt->bind_param("ssss", $funame,$luname,$email,$upswd);
      $stmt->execute();
      if ($_SERVER["REQUEST_METHOD"] == "POST") {
        $funame = $_POST["funame"];
        $upswd = $_POST["upswd"];
      echo "New record inserted sucessfully";
      header("Location:login.html");
      exit();
     } else {
      echo "Someone already register using this email";

     }
     $stmt->close();
     $conn->close();
    }
} else {
 echo "All field are required";
 die();
}
}
}
?>